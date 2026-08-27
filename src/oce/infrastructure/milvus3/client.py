"""Async Milvus client for dense vector retrieval."""

import asyncio
from typing import Any

from loguru import logger
from pymilvus import AsyncMilvusClient, MilvusClient
from pymilvus.client.types import LoadState

from oce.shared.config.settings import MilvusSettings

from .schema import create_oce_collection_schema

_MAX_CONTENT_BYTES = 65_535


def _validate_blob_name(blob_name: str) -> str:
    if len(blob_name) != 64 or any(
            character not in "0123456789abcdef" for character in blob_name.casefold()
    ):
        raise ValueError("Milvus blob filters require a SHA256 blob_name")
    return blob_name


def _fit_content_field(content: str) -> tuple[str, bool]:
    """Fit text into Milvus VARCHAR without splitting a UTF-8 code point."""
    encoded = content.encode("utf-8")
    if len(encoded) <= _MAX_CONTENT_BYTES:
        return content, False
    return encoded[:_MAX_CONTENT_BYTES].decode("utf-8", errors="ignore"), True


class Milvus3Client:
    """Manage the OCE collection and its dense vector search operations."""

    def __init__(self, settings: MilvusSettings):
        """初始化远程异步或本地同步 Milvus 客户端。

        Args:
            settings: Milvus 配置
        """
        self.settings = settings
        self._local = not settings.endpoint.startswith(("http://", "https://"))
        client_type = MilvusClient if self._local else AsyncMilvusClient
        self._client = client_type(uri=settings.endpoint, token=settings.token)
        self._initialize_lock = asyncio.Lock()
        self._initialized = False
        logger.info("Connecting to Milvus: {}", settings.endpoint)

    async def _call(self, method_name: str, *args, **kwargs):
        """统一异步调用远程客户端和本地 Lite 同步客户端。"""
        method = getattr(self._client, method_name)
        if self._local:
            return await asyncio.to_thread(method, *args, **kwargs)
        return await method(*args, **kwargs)

    async def initialize(self):
        """异步初始化：确保 Collection 存在"""
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            await self._ensure_collection()
            self._initialized = True

    async def _ensure_collection(self):
        """确保 Collection 存在（自动创建索引）"""
        collection_name = self.settings.collection_name

        if await self._call("has_collection", collection_name):
            logger.info("Milvus collection exists: {}", collection_name)
            # get_load_state 返回 {"state": LoadState}；Milvus Lite 的 load 状态不跨进程持久，
            # 重启后 collection 回到 released，必须重新 load 才能 search（load_collection 幂等）
            state = await self._call("get_load_state", collection_name)
            load_state = state.get("state") if isinstance(state, dict) else state
            if load_state != LoadState.Loaded:
                logger.info("Loading Milvus collection: {}", collection_name)
                await self._call("load_collection", collection_name)
            return

        logger.info("Creating Milvus collection: {}", collection_name)

        # 创建 Schema
        schema = create_oce_collection_schema(
            dense_dim=self.settings.dense_dim,
        )

        # 先创建 Collection（不带索引）
        await self._call(
            "create_collection",
            collection_name=collection_name,
            schema=schema,
        )

        # 再异步创建索引（Milvus Lite 可能不支持同步等待）
        index_params = self._build_index_params()
        try:
            await self._call(
                "create_index",
                collection_name=collection_name,
                index_params=index_params,
            )
            logger.success("Collection indexes created")
        except Exception as exc:
            if self.settings.endpoint.startswith("http"):
                raise RuntimeError(
                    f"Failed to create Milvus indexes for {collection_name}"
                ) from exc
            logger.warning("Milvus Lite did not create all indexes: {}", exc)

        # 加载到内存
        await self._call("load_collection", collection_name)
        logger.success("Milvus collection ready: {}", collection_name)

    def _build_index_params(self):
        """构建索引参数（使用 client.prepare_index_params）"""
        index_params = self._client.prepare_index_params()

        # 密集向量索引（HNSW）
        index_params.add_index(
            field_name="dense_vector",
            index_type=self.settings.dense_index_type,
            metric_type=self.settings.dense_metric_type,
            params={
                "M": self.settings.hnsw_m,
                "efConstruction": self.settings.hnsw_ef_construction,
            },
        )

        return index_params

    async def insert(self, chunks: list[dict[str, Any]]) -> dict[str, Any]:
        """插入向量

        Args:
            chunks: 代码块列表，每个元素包含:
                - chunk_id: str
                - content_hash: str
                - content: str
                - embedding: list[float]  # dense_vector
                - blob_name: str
                - metadata: dict (path/start_line/end_line)

        Returns:
            插入结果统计
        """
        if not chunks:
            return {"inserted": 0}

        data = []
        for chunk in chunks:
            content, truncated = _fit_content_field(chunk["content"])
            metadata = dict(chunk.get("metadata", {}))
            if truncated:
                metadata["content_truncated"] = True
                metadata["content_bytes"] = len(chunk["content"].encode("utf-8"))
            data.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "content_hash": chunk["content_hash"],
                    "content": content,
                    "dense_vector": chunk["embedding"],
                    "blob_name": chunk["blob_name"],
                    "metadata": metadata,
                }
            )

        result = await self._call(
            "upsert",
            collection_name=self.settings.collection_name,
            data=data,
        )

        count = result.get("upsert_count", result.get("insert_count", len(data)))
        logger.info("Upserted {} vectors", count)
        return {"inserted": count}

    async def search(
        self,
        query_text: str,
        query_embedding: list[float],
        blob_filter: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """执行 Milvus dense 向量检索。"""
        filter_expr = None
        if blob_filter:
            blob_list = ", ".join(
                f'"{_validate_blob_name(blob_name)}"' for blob_name in blob_filter
            )
            filter_expr = f"blob_name in [{blob_list}]"

        dense_limit = top_k * 2
        dense_ef = max(self.settings.hnsw_ef_search, dense_limit)
        try:
            results = await self._call(
                "search",
                collection_name=self.settings.collection_name,
                data=[query_embedding],
                anns_field="dense_vector",
                search_params={
                    "metric_type": self.settings.dense_metric_type,
                    "params": {"ef": dense_ef},
                },
                limit=top_k,
                filter=filter_expr,
                output_fields=["chunk_id", "content_hash", "content", "blob_name", "metadata"],
            )
        except Exception as exc:
            logger.error(
                "Milvus dense search failed: query_text_len={}, embedding_dim={}, top_k={}, error={}",
                len(query_text),
                len(query_embedding),
                top_k,
                exc,
            )
            raise

        if not results:
            logger.warning("Dense search returned no result groups")
            return []

        hits = results[0]
        formatted: list[dict[str, Any]] = []
        for hit in hits:
            entity = hit.get("entity", hit) if isinstance(hit, dict) else hit.entity
            score = hit.get("distance", hit.get("score")) if isinstance(hit, dict) else hit.distance
            formatted.append(
                {
                    "content_hash": entity.get("content_hash"),
                    "content": entity.get("content"),
                    "blob_name": entity.get("blob_name"),
                    "metadata": entity.get("metadata"),
                    "score": score,
                }
            )
        return formatted

    async def delete_by_blob(self, blob_name: str) -> int:
        """删除指定 Blob 的所有向量

        Args:
            blob_name: Blob 名称

        Returns:
            删除的向量数
        """
        _validate_blob_name(blob_name)
        result = await self._call(
            "delete",
            collection_name=self.settings.collection_name,
            filter=f'blob_name == "{blob_name}"',
        )

        # Milvus 3.0 delete 返回 list（删除的主键列表）
        deleted = len(result) if isinstance(result, list) else 0
        logger.info("Deleted {} vectors for blob {}", deleted, blob_name)
        return deleted

    async def close(self):
        """关闭客户端连接"""
        await self._call("close")
        logger.info("Milvus client closed")
