"""路径索引 - 专用于文件名查询的向量索引"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
)

from oce.domain.services.path_search import PathSearchResult, PathSearchStore
from oce.shared.config.settings import MilvusSettings


class PathIndexClient:
    """路径索引客户端 - 只存储路径信息的轻量索引"""

    def __init__(self, settings: MilvusSettings):
        self.collection = None
        self.settings = settings
        self.collection_name = settings.path_collection_name
        self.dense_dim = settings.dense_dim  # 从配置读取维度
        self._initialized = False

    async def initialize(self) -> None:
        """初始化连接和集合"""
        if self._initialized:
            return

        # 连接 Milvus
        connections.connect(
            alias="default",
            uri=self.settings.endpoint,
            token=self.settings.token,
        )

        # 创建或加载集合
        if not self._collection_exists():
            self._create_collection()
        
        self.collection = Collection(self.collection_name)
        self.collection.load()
        
        self._initialized = True
        logger.info(f"PathIndexClient initialized, collection: {self.collection_name}")

    def _collection_exists(self) -> bool:
        """检查集合是否存在"""
        from pymilvus import utility
        return utility.has_collection(self.collection_name)

    def _create_collection(self) -> None:
        """创建路径索引集合"""
        fields = [
            FieldSchema(
                name="path_id",
                dtype=DataType.VARCHAR,
                max_length=512,
                is_primary=True,
                description="path_{blob_name}",
            ),
            FieldSchema(
                name="blob_name",
                dtype=DataType.VARCHAR,
                max_length=64,
                description="Blob identifier for filtering",
            ),
            FieldSchema(
                name="path",
                dtype=DataType.VARCHAR,
                max_length=512,
                description="Original file path",
            ),
            FieldSchema(
                name="path_document",
                dtype=DataType.VARCHAR,
                max_length=2048,
                description="Rich semantic path document",
            ),
            FieldSchema(
                name="path_vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.dense_dim,  # 使用配置的维度
                description="Path document embedding",
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Path-only index for filename queries",
        )

        collection = Collection(name=self.collection_name, schema=schema)

        # 创建 HNSW 索引
        index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {"M": 16, "efConstruction": 256},
        }
        collection.create_index(field_name="path_vector", index_params=index_params)

        logger.info(f"Created path index collection: {self.collection_name}")

    async def insert(self, path_docs: list[dict[str, Any]]) -> dict[str, Any]:
        """
        插入路径文档
        
        Args:
            path_docs: 路径文档列表，每个包含:
                - path_id: str (e.g., "path_{blob_name}")
                - blob_name: str
                - path: str
                - path_document: str
                - path_vector: list[float]
        
        Returns:
            插入结果统计
        """
        if not path_docs:
            return {"inserted": 0}

        await self.initialize()

        # 准备数据
        data = []
        for doc in path_docs:
            data.append({
                "path_id": doc["path_id"],
                "blob_name": doc["blob_name"],
                "path": doc["path"],
                "path_document": doc["path_document"],
                "path_vector": doc["path_vector"],
            })

        # 分批 upsert：单次请求受 Milvus gRPC 消息体上限约束，整仓上万条向量必须切片
        batch_size = 1000
        count = 0
        for start in range(0, len(data), batch_size):
            chunk = data[start:start + batch_size]
            result = self.collection.upsert(chunk)
            count += result.upsert_count if hasattr(result, "upsert_count") else len(chunk)
        
        logger.info(f"Upserted {count} path documents")
        return {"inserted": count}

    async def search_paths(
        self,
        query_vector: list[float],
        allowed_blob_names: list[str] | None = None,
        top_k: int = 20,
    ) -> list[PathSearchResult]:
        """
        搜索路径索引（实现 PathSearchStore Protocol）

        Args:
            query_vector: 查询向量
            allowed_blob_names: 允许的 blob 过滤
            top_k: 返回结果数

        Returns:
            路径搜索结果列表
        """
        await self.initialize()

        # 构建过滤表达式
        filter_expr = None
        if allowed_blob_names:
            blob_list = ", ".join(f'"{name}"' for name in allowed_blob_names)
            filter_expr = f"blob_name in [{blob_list}]"

        # 执行搜索
        search_params = {
            "metric_type": "COSINE",
            "params": {"ef": max(64, top_k * 2)},
        }

        results = self.collection.search(
            data=[query_vector],
            anns_field="path_vector",
            param=search_params,
            limit=top_k,
            expr=filter_expr,
            output_fields=["blob_name", "path"],
        )

        # 转换为 PathSearchResult
        hits = []
        if results and len(results) > 0:
            for result in results[0]:
                hits.append(
                    PathSearchResult(
                        path=result.entity.get("path"),
                        blob_name=result.entity.get("blob_name"),
                        score=float(result.score),
                    )
                )

        logger.debug(f"Path index search returned {len(hits)} results")
        return hits

    async def delete_by_blob_names(self, blob_names: list[str]) -> None:
        """删除指定 blob 的路径文档"""
        await self.initialize()
        
        expr = f"blob_name in [{', '.join(f'"{name}"' for name in blob_names)}]"
        self.collection.delete(expr)
        logger.info(f"Deleted path documents for {len(blob_names)} blobs")

    async def get_stats(self) -> dict[str, Any]:
        """获取索引统计信息"""
        await self.initialize()
        
        stats = self.collection.num_entities
        return {
            "collection_name": self.collection_name,
            "total_paths": stats,
        }
