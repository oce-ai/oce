"""Milvus 3.0 SearchStore 实现

实现 domain/services/search.py 的 SearchStore Protocol：
- dense 向量检索
- 索引级 blob 过滤（多租户关键）
- 返回 SearchHit 值对象列表
"""

from __future__ import annotations

from typing import Any, Sequence

from oce.domain.services.search import SearchHit
from oce.shared.config.settings import MilvusSettings

from .client import Milvus3Client


class Milvus3SearchStore:
    """Milvus 3.0 SearchStore（实现 SearchStore Protocol）"""

    def __init__(self, milvus_settings: MilvusSettings):
        self.client = Milvus3Client(milvus_settings)
        self.milvus_settings = milvus_settings
        self._initialized = False

    async def _ensure_initialized(self):
        """确保客户端已初始化"""
        if not self._initialized:
            await self.client.initialize()
            self._initialized = True

    async def search(
        self,
        *,
        query: str,
        query_vector: list[float],
        allowed_blob_names: Sequence[str] | None = None,
        top_k: int = 50,
        vector_threshold: float = 0.1,
    ) -> list[SearchHit]:
        """执行 dense 向量检索，返回 SearchHit 列表。"""
        if allowed_blob_names is not None and not allowed_blob_names:
            return []
        await self._ensure_initialized()

        results = await self.client.search(
            query_text=query,
            query_embedding=query_vector,
            blob_filter=(
                list(allowed_blob_names) if allowed_blob_names is not None else None
            ),
            top_k=top_k,
        )
        hits: list[SearchHit] = []
        for result in results:
            score = result.get("score", 0.0)
            if score < vector_threshold:
                continue
            metadata = result.get("metadata", {})
            hits.append(SearchHit(
                blob_name=result.get("blob_name", ""),
                path=metadata.get("path", result.get("blob_name", "")),
                content=result.get("content", ""),
                score=score,
                content_hash=result.get("content_hash", ""),
                start_line=metadata.get("start_line", 1),
                end_line=metadata.get("end_line", 1),
            ))
        return hits

    async def upsert(self, items: list[dict[str, Any]]) -> None:
        """写入向量到 Milvus

        Args:
            items: 每个包含 blob_name, chunk_id, vector, content, metadata
        """
        await self._ensure_initialized()

        chunks = [
            {
                "chunk_id": item["chunk_id"],
                "content_hash": item.get("content_hash", item["chunk_id"]),
                "content": item.get("content", ""),
                "embedding": item["vector"],
                "blob_name": item["blob_name"],
                "metadata": item.get("metadata", {}),
            }
            for item in items
        ]
        await self.client.insert(chunks)

    async def delete(self, blob_names: list[str]) -> None:
        """删除指定 blob 的全部向量"""
        await self._ensure_initialized()
        for blob_name in blob_names:
            await self.client.delete_by_blob(blob_name)

    async def close(self):
        """关闭客户端连接"""
        await self.client.close()
