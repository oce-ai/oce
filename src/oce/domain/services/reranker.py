"""Reranker 领域服务 - 召回精排

解决「单篇多相关」：对 store 召回的结果做二次精排（跨文档比较）。
NoopReranker 用于关闭重排（起步阶段退化为纯召回排序）。
"""

from __future__ import annotations

from typing import Protocol

from oce.domain.services.search import SearchHit


class Reranker(Protocol):
    """重排器协议"""

    async def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        """对召回结果精排，返回重排后的列表"""
        ...


class NoopReranker:
    """不重排（原样返回）"""

    async def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        return hits
