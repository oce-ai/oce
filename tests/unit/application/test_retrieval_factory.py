"""检索栈工厂等价性测试。

守住 Commit 1 的零行为变更：``build_retrieval_stack`` 的产出必须与原 Container 内联
构造逐点对应。重点覆盖那些「构造期固化」的值——它们正是热重载必须重建 pipeline 的原因，
也是最容易在抽取时丢失的地方。

全部用 fakes + 受控 Settings 快照，不连 DB / Milvus / Redis。
"""

from __future__ import annotations

import pytest

from oce.application.factories.retrieval import RetrievalDeps, build_retrieval_stack
from oce.application.queries.search import SearchQueryHandler
from oce.domain.services.retrieval import RetrievalPipeline
from oce.domain.services.reranker import NoopReranker
from oce.domain.services.selector.coverage_selector import CoverageSelector
from oce.infrastructure.persistence.symbol_search_store import SymbolSearchStore
from oce.shared.config.settings import LLMSettings, Settings
from oce.shared.metrics import NoopMetricsSink

from tests.unit.application.fakes import FakeEmbedder, FakeSearchStore


class _FakeReranker:
    """替身 base reranker；工厂应原样把它作为 pipeline.reranker 传入。"""

    async def rerank(self, query, hits):
        return hits


class _FakePathIndex:
    """替身 PathIndexClient。"""


def _settings(**retrieval_overrides) -> Settings:
    """一份受控 Settings 快照；retrieval 组按需覆盖，LLM 组全关。

    注意 retrieval 组默认 ``intent_classification_enabled=True``，故这里显式关闭，
    让「flags 全关」的用例名副其实。
    """
    retrieval_overrides.setdefault("intent_classification_enabled", False)
    base = Settings(
        llm=LLMSettings(
            rerank_enabled=False, api_key="k", model="m", base_url="http://x"
        ),
    )
    retrieval = base.retrieval.model_copy(update=retrieval_overrides)
    return base.model_copy(update={"retrieval": retrieval})


def _deps(**overrides) -> RetrievalDeps:
    defaults = dict(
        embedder=FakeEmbedder(),
        search_store=FakeSearchStore(),
        base_reranker=_FakeReranker(),
        path_index=_FakePathIndex(),
        session_factory=lambda: None,
        metrics=NoopMetricsSink(),
        retrieval_audit_enabled=False,
        store_query_text=False,
        token_usage_cb=None,
        llm_settings=LLMSettings(
            rerank_enabled=False, api_key="k", model="m", base_url="http://x"
        ),
    )
    defaults.update(overrides)
    return RetrievalDeps(**defaults)


class TestSelectorConstruction:
    def test_selector_uses_snapshot_chunk_budget(self):
        """max_chunks_per_path / max_context_chars / overlap_threshold 是构造期固化的，
        必须来自传入的快照而非全局 get_settings()。"""
        settings = _settings(
            max_chunks_per_path=5, max_context_chars=8_000, overlap_threshold=0.25
        )
        stack = build_retrieval_stack(settings, _deps())

        selector = stack.pipeline.selector
        assert isinstance(selector, CoverageSelector)
        assert selector.max_per_path == 5
        assert selector.max_chars == 8_000
        assert selector.overlap_threshold == 0.25

    def test_pipeline_holds_snapshot_settings_by_reference(self):
        settings = _settings(default_top_k=42)
        stack = build_retrieval_stack(settings, _deps())

        assert stack.pipeline.settings is settings.retrieval


class TestSymbolStoreRebuild:
    def test_symbol_store_captures_scope_and_timeout(self):
        """exact_max_scope_blobs / exact_timeout_seconds 是双份捕获：pipeline 调用时读
        一份，SymbolSearchStore 构造时又固化一份。工厂必须用快照值重建后者。"""
        settings = _settings(exact_max_scope_blobs=777, exact_timeout_seconds=3.5)
        stack = build_retrieval_stack(settings, _deps())

        assert isinstance(stack.symbol_store, SymbolSearchStore)
        assert stack.symbol_store._max_scope_blobs == 777
        assert stack.symbol_store._timeout_seconds == 3.5
        # pipeline 持有的就是这份重建过的 store
        assert stack.pipeline.exact_store is stack.symbol_store


class TestComponentGating:
    def test_path_store_mounted_when_enabled(self):
        deps = _deps()
        settings = _settings(path_index_enabled=True)
        stack = build_retrieval_stack(settings, deps)

        assert stack.pipeline.path_store is deps.path_index

    def test_path_store_detached_when_disabled(self):
        deps = _deps()
        settings = _settings(path_index_enabled=False)
        stack = build_retrieval_stack(settings, deps)

        assert stack.pipeline.path_store is None

    def test_base_reranker_always_passed_through(self):
        """reranker 不按 enabled 开关：始终传 deps.base_reranker，enabled 语义在其内部
        _resolve_config。在此换成 Noop 会改变冷启动行为。"""
        deps = _deps()
        stack = build_retrieval_stack(_settings(), deps)

        assert stack.pipeline.reranker is deps.base_reranker
        assert not isinstance(stack.pipeline.reranker, NoopReranker)

    def test_llm_components_absent_when_flags_off(self):
        stack = build_retrieval_stack(_settings(), _deps())

        assert stack.pipeline.llm_reranker is None
        assert stack.pipeline.query_rewriter is None
        assert stack.pipeline.intent_classifier is None
        # 全关 → 不构造任何 LLM client（避免 /admin/credentials/reload 多出无谓 DB 查询）
        assert stack.llm_clients == ()

    def test_query_rewriter_built_when_enabled(self):
        settings = _settings(query_rewrite_enabled=True)
        stack = build_retrieval_stack(settings, _deps())

        assert stack.pipeline.query_rewriter is not None
        assert len(stack.llm_clients) == 1
        assert stack.llm_clients[0]._kind == "query_rewrite"

    def test_llm_reranker_built_when_enabled(self):
        settings = Settings(
            llm=LLMSettings(
                rerank_enabled=True, api_key="k", model="m", base_url="http://x",
                max_candidates=33, output_top_k=7, snippet_chars=500,
            ),
        )
        stack = build_retrieval_stack(settings, _deps(llm_settings=settings.llm))

        reranker = stack.pipeline.llm_reranker
        assert reranker is not None
        # LLMReranker 的参数全是构造期固化，必须来自快照
        assert reranker.max_candidates == 33
        assert reranker.output_top_k == 7
        assert reranker.snippet_chars == 500
        assert stack.llm_clients[0]._kind == "llm_rerank"


class TestHandlerWiring:
    def test_handler_wraps_pipeline_with_audit_flags(self):
        deps = _deps(retrieval_audit_enabled=True, store_query_text=True)
        stack = build_retrieval_stack(_settings(), deps)

        assert isinstance(stack.handler, SearchQueryHandler)
        assert stack.handler.pipeline is stack.pipeline
        assert stack.handler.retrieval_audit_enabled is True
        assert stack.handler.store_query_text is True

    def test_handler_is_pipeline_instance(self):
        stack = build_retrieval_stack(_settings(), _deps())
        assert isinstance(stack.pipeline, RetrievalPipeline)


class TestPurity:
    def test_repeated_builds_are_independent(self):
        """工厂可被反复调用且互不干扰：每次产出全新的 pipeline / handler / store。
        这是热重载「构造成功才生效」原子性的前提。"""
        deps = _deps()
        settings = _settings(default_top_k=10)
        a = build_retrieval_stack(settings, deps)
        b = build_retrieval_stack(settings, deps)

        assert a.pipeline is not b.pipeline
        assert a.handler is not b.handler
        assert a.symbol_store is not b.symbol_store
        # 复用的昂贵依赖保持同一实例
        assert a.pipeline.embedder is b.pipeline.embedder is deps.embedder

    def test_construction_failure_leaves_nothing(self):
        """构造期抛错 → 调用方拿不到 stack，不会半生效。用非法 selector 参数触发
        CoverageSelector 的校验。"""
        settings = _settings(max_chunks_per_path=0)  # CoverageSelector 要求 >=1
        with pytest.raises(ValueError):
            build_retrieval_stack(settings, _deps())
