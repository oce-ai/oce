"""检索审计：pipeline 阶段打点填充 audit + handler 按 source 上报（含空回、开关）。"""

from __future__ import annotations

from oce.application.queries.search import SearchQuery, SearchQueryHandler
from oce.domain.services.retrieval import RetrievalPipeline
from oce.domain.services.search import SearchHit
from oce.shared.metrics import RetrievalAudit, RetrievalMetricRecord

from tests.unit.application.fakes import FakeEmbedder, FakeSearchStore


class RecordingSink:
    """只捕获 retrieval 上报的测试替身。"""

    def __init__(self) -> None:
        self.retrieval: list[RetrievalMetricRecord] = []

    def record_retrieval(self, record: RetrievalMetricRecord) -> None:
        self.retrieval.append(record)


def _hit() -> SearchHit:
    return SearchHit(
        blob_name="h1",
        path="src/main.py",
        content="def main(): pass",
        score=0.9,
        start_line=1,
        end_line=1,
    )


def _pipeline(hits: list[SearchHit]) -> RetrievalPipeline:
    return RetrievalPipeline(embedder=FakeEmbedder(), store=FakeSearchStore(hits=hits))


def _handler(hits, **kwargs) -> tuple[SearchQueryHandler, RecordingSink]:
    sink = RecordingSink()
    handler = SearchQueryHandler(_pipeline(hits), metrics=sink, **kwargs)
    return handler, sink


class TestPipelineAuditFill:
    async def test_stages_and_scope_filled(self):
        audit = RetrievalAudit()
        await _pipeline([_hit()]).search(
            "main entry", frozenset({"h1", "h2"}), audit=audit
        )

        # 无 intent 分类器/改写器 → 只跑核心阶段；stage() 应填充这些键
        assert "dense" in audit.stages
        assert "select" in audit.stages
        assert all(v >= 0 for v in audit.stages.values())
        assert audit.scope_size == 2
        assert audit.intent is None
        assert audit.path_boosted is False

    async def test_audit_none_is_zero_overhead(self):
        # 不传 audit：不打点、不报错，行为与原来一致
        hits = await _pipeline([_hit()]).search("main entry")
        assert len(hits) == 1


class TestHandlerReporting:
    async def test_reports_source_and_hit_count(self):
        handler, sink = _handler([_hit()], retrieval_audit_enabled=True)
        await handler.handle(
            SearchQuery("main entry", frozenset({"h1"}), source="overview")
        )

        assert len(sink.retrieval) == 1
        rec = sink.retrieval[0]
        assert rec.source == "overview"
        assert rec.hit_count == 1
        assert rec.total_ms >= 0
        assert "select" in rec.stages

    async def test_empty_return_recorded_as_zero(self):
        handler, sink = _handler([], retrieval_audit_enabled=True)
        result = await handler.handle(SearchQuery("main entry", frozenset({"h1"})))

        assert result.hits == []
        assert len(sink.retrieval) == 1
        assert sink.retrieval[0].hit_count == 0  # 空回

    async def test_disabled_does_not_report(self):
        handler, sink = _handler([_hit()], retrieval_audit_enabled=False)
        result = await handler.handle(SearchQuery("main entry"))

        assert len(result.hits) == 1
        assert sink.retrieval == []  # 关闭时完全不上报

    async def test_query_text_switch(self):
        on_handler, on_sink = _handler(
            [_hit()], retrieval_audit_enabled=True, store_query_text=True
        )
        off_handler, off_sink = _handler(
            [_hit()], retrieval_audit_enabled=True, store_query_text=False
        )
        await on_handler.handle(SearchQuery("secret query", frozenset({"h1"})))
        await off_handler.handle(SearchQuery("secret query", frozenset({"h1"})))

        assert on_sink.retrieval[0].query_text == "secret query"
        assert off_sink.retrieval[0].query_text is None  # 默认不留存原文
