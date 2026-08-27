"""路径搜索接口 - 文件名查询专用索引"""

from __future__ import annotations

from typing import Any, Protocol


class PathSearchResult:
    """路径搜索结果"""

    def __init__(
        self,
        path: str,
        blob_name: str,
        score: float,
    ):
        self.path = path
        self.blob_name = blob_name
        self.score = score


class PathSearchStore(Protocol):
    """路径搜索存储协议 - 检索 + 写入路径索引"""

    async def search_paths(
        self,
        query_vector: list[float],
        allowed_blob_names: list[str] | None = None,
        top_k: int = 20,
    ) -> list[PathSearchResult]:
        """
        路径向量检索

        Args:
            query_vector: 查询向量
            allowed_blob_names: 允许的 blob 过滤
            top_k: 返回数量

        Returns:
            路径搜索结果列表
        """
        ...

    async def insert(self, path_docs: list[dict[str, Any]]) -> dict[str, Any]:
        """写入路径文档。

        Args:
            path_docs: 每个元素含 path_id / blob_name / path / path_document / path_vector。

        Returns:
            插入统计（如 {"inserted": n}）。
        """
        ...

    async def delete_by_blob_names(self, blob_names: list[str]) -> None:
        """删除指定 blob 的路径文档。"""
        ...
