"""查询对象与处理器 - 检索读路径

SearchQuery: 一次代码检索（向量召回 + 精确标识符召回 + 重排 + 覆盖度选择）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from oce.application.messages import Query
from oce.domain.services.retrieval import RetrievalPipeline
from oce.domain.services.search import SearchHit


@dataclass(frozen=True)
class SearchQuery(Query):
    """检索查询"""

    query: str
    allowed_blob_names: frozenset[str] | None = None


@dataclass(frozen=True)
class SearchResult:
    """检索结果"""

    hits: list[SearchHit] = field(default_factory=list)


class SearchQueryHandler:
    """处理 SearchQuery"""

    def __init__(self, pipeline: RetrievalPipeline) -> None:
        self.pipeline = pipeline

    async def handle(self, query: SearchQuery) -> SearchResult:
        hits = await self.pipeline.search(query.query, query.allowed_blob_names)
        return SearchResult(hits=hits)
