"""查询对象与处理器 - 检索读路径

SearchQuery: 一次代码检索（向量召回 + 精确标识符召回 + 重排 + 覆盖度选择）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from oce.application.messages import Query
from oce.domain.services.retrieval import RetrievalPipeline
from oce.domain.services.search import SearchHit
from oce.shared.metrics import (
    MetricsSink,
    NoopMetricsSink,
    RetrievalAudit,
    RetrievalMetricRecord,
)


@dataclass(frozen=True)
class SearchQuery(Query):
    """检索查询"""

    query: str
    allowed_blob_names: frozenset[str] | None = None
    source: str = "retrieval"


@dataclass(frozen=True)
class SearchResult:
    """检索结果"""

    hits: list[SearchHit] = field(default_factory=list)


class SearchQueryHandler:
    """处理 SearchQuery。

    检索审计开启时，为本次检索创建 RetrievalAudit 传入 pipeline 收集各阶段耗时，
    检索完成后按 source 上报（hit_count=0 即空回）。审计上报走旁路 sink，不影响主链路。
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        *,
        metrics: MetricsSink | None = None,
        retrieval_audit_enabled: bool = False,
        store_query_text: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.metrics = metrics or NoopMetricsSink()
        self.retrieval_audit_enabled = retrieval_audit_enabled
        self.store_query_text = store_query_text

    async def handle(self, query: SearchQuery) -> SearchResult:
        if not self.retrieval_audit_enabled:
            hits = await self.pipeline.search(query.query, query.allowed_blob_names)
            return SearchResult(hits=hits)

        audit = RetrievalAudit()
        started = perf_counter()
        hits = await self.pipeline.search(
            query.query, query.allowed_blob_names, audit=audit
        )
        total_ms = int((perf_counter() - started) * 1000)
        self.metrics.record_retrieval(
            RetrievalMetricRecord(
                source=query.source,
                hit_count=len(hits),
                total_ms=total_ms,
                scope_size=audit.scope_size,
                intent=audit.intent,
                path_boosted=audit.path_boosted,
                query_text=query.query if self.store_query_text else None,
                stages=dict(audit.stages),
            )
        )
        return SearchResult(hits=hits)
