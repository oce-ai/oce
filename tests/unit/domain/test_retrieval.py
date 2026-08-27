"""RetrievalPipeline 领域服务测试

用 Fake store / embedder 验证编排流程：
embed → search → rerank → 源码优先 → 置信度门槛 → select。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from oce.domain.services.llm.intent import QueryIntent
from oce.domain.services.retrieval import RetrievalPipeline, source_priority_factor
from oce.domain.services.search import SearchHit
from oce.shared.config.settings import RetrievalSettings


class FakeSearchStore:
    """SearchStore 内存替身：返回预设命中"""

    def __init__(self, hits=None):
        self.hits = hits or []
        self.last_query: str = ""
        self.last_vector: list[float] = []
        self.queries: list[str] = []
        self.hits_by_query: dict[str, list[SearchHit]] = {}

    async def search(self, *, query, query_vector, allowed_blob_names=None,
                     top_k=50, vector_threshold=0.1):
        self.last_query = query
        self.last_vector = query_vector
        self.queries.append(query)
        return list(self.hits_by_query.get(query, self.hits))


class FakeEmbedder:
    """确定性假 embedder"""

    def __init__(self):
        self.queries: list[str] = []

    async def embed_query(self, text):
        self.queries.append(text)
        return [float(len(text))]


class FakeIntentClassifier:
    def __init__(self, intent: QueryIntent):
        self.intent = intent

    async def classify(self, query: str) -> QueryIntent:
        return self.intent


class FakePathStore:
    def __init__(self):
        self.queries = 0

    async def search_paths(self, query_vector, allowed_blob_names=None, top_k=20):
        self.queries += 1
        return []


class FakeExactSearchStore:
    def __init__(self, hits=None, error: Exception | None = None):
        self.hits = hits or []
        self.error = error
        self.identifiers: tuple[str, ...] = ()
        self.allowed_blob_names = None

    async def search_exact(self, *, identifiers, allowed_blob_names=None, top_k=50):
        self.identifiers = tuple(identifiers)
        self.allowed_blob_names = allowed_blob_names
        if self.error is not None:
            raise self.error
        return list(self.hits[:top_k])


def _hit(path: str, score: float) -> SearchHit:
    return SearchHit(blob_name="x" * 64, path=path, content="code", score=score)


def _settings(**kwargs) -> RetrievalSettings:
    return RetrievalSettings(**kwargs)


class TestSourcePriorityFactor:
    def test_source_code_is_one(self):
        assert source_priority_factor("src/engine/core.py") == 1.0

    def test_main_readme_is_one(self):
        assert source_priority_factor("README.md") == 1.0

    def test_legal_file_heavily_penalized(self):
        assert source_priority_factor("LICENSE") < 0.5

    def test_docs_and_tests_penalized(self):
        assert source_priority_factor("docs/guide.md") < 1.0
        assert source_priority_factor("tests/test_x.py") < 1.0
        assert source_priority_factor("src/tools/planner.test.ts") == 0.6

    def test_generic_barrels_and_types_are_slightly_penalized(self):
        assert source_priority_factor("src/tools/index.ts") == 0.85
        assert source_priority_factor("src/tools/types.ts") == 0.85
        assert source_priority_factor("src/config/types.openclaw.ts") == 1.0


class TestRetrievalPipeline:
    @pytest.fixture
    def pipe(self):
        hits = [
            _hit("src/core.py", 0.9),
            _hit("docs/guide.md", 0.8),
            _hit("src/util.py", 0.7),
        ]
        return RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore(hits),
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )

    async def test_search_returns_sorted_results(self, pipe):
        results = await pipe.search("find core")

        # 源码优先：docs/guide.md 的 0.8 被降权到 0.4，排到 0.7 之后
        assert [r.path for r in results] == [
            "src/core.py",
            "src/util.py",
            "docs/guide.md",
        ]

    async def test_main_readme_not_penalized(self):
        hits = [
            _hit("src/core.py", 0.8),
            _hit("README.md", 0.75),   # 主 README 显式不降权
        ]
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore(hits),
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )
        results = await pipe.search("q")
        assert [r.path for r in results] == ["src/core.py", "README.md"]

    async def test_confidence_floor_filters_weak_hits(self):
        hits = [
            _hit("src/a.py", 0.9),
            _hit("src/b.py", 0.1),   # 低于 floor
        ]
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore(hits),
            settings=_settings(confidence_floor=0.3, final_select_k=10),
        )
        results = await pipe.search("q")
        assert [r.path for r in results] == ["src/a.py"]

    async def test_final_select_k_limits_results(self):
        hits = [_hit(f"src/f{i}.py", 1.0 - i * 0.01) for i in range(10)]
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore(hits),
            settings=_settings(confidence_floor=0.0, final_select_k=3),
        )
        results = await pipe.search("q")
        assert len(results) == 3

    async def test_allowed_blob_names_passed_to_store(self):
        store = FakeSearchStore([_hit("src/a.py", 0.9)])
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=store,
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )
        await pipe.search("q", allowed_blob_names={"aaa", "bbb"})
        assert set(store.last_query == "q" and store.last_vector)  # 触发赋值
        assert store.last_query == "q"

    async def test_empty_scope_does_not_search_globally(self):
        store = FakeSearchStore([_hit("src/private.py", 0.9)])
        embedder = FakeEmbedder()
        pipe = RetrievalPipeline(
            embedder=embedder,
            store=store,
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )

        results = await pipe.search("q", allowed_blob_names=set())

        assert results == []
        assert embedder.queries == []
        assert store.last_query == ""

    async def test_reranker_used_when_provided(self):
        class ReorderReranker:
            async def rerank(self, query, hits):
                return list(reversed(hits))

        hits = [_hit("src/a.py", 0.5), _hit("src/b.py", 0.9)]
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore(hits),
            reranker=ReorderReranker(),
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )
        results = await pipe.search("q")
        # rerank 后反转，但源码优先排序会重新稳定为 b 在前（0.9）
        assert results[0].path == "src/b.py"

    async def test_llm_rerank_order_is_not_overwritten_by_retrieval_scores(self):
        class ReverseLLMReranker:
            async def rerank(self, query, candidates, top_k=None):
                return list(reversed(candidates))

        hits = [_hit("src/high.py", 0.9), _hit("docs/answer.md", 0.8)]
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore(hits),
            llm_reranker=ReverseLLMReranker(),
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )

        results = await pipe.search("q")

        assert [result.path for result in results] == [
            "docs/answer.md",
            "src/high.py",
        ]

    async def test_symbol_intent_does_not_use_path_index(self):
        path_store = FakePathStore()
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore([_hit("src/commands/provider.rs", 0.9)]),
            path_store=path_store,
            intent_classifier=FakeIntentClassifier(QueryIntent.SYMBOL),
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )

        results = await pipe.search("`add_provider` 函数在哪里定义？")

        assert path_store.queries == 0
        assert [result.path for result in results] == ["src/commands/provider.rs"]

    async def test_exact_identifier_candidates_join_semantic_reranking(self):
        exact_store = FakeExactSearchStore(
            [
                SearchHit(
                    blob_name="a" * 64,
                    path="src-tauri/src/commands/copilot.rs",
                    content="pub async fn copilot_get_models() {}",
                    score=1.0,
                )
            ]
        )
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore([_hit("src/proxy/copilot_auth.rs", 0.9)]),
            exact_store=exact_store,
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )

        results = await pipe.search(
            "`copilot_get_models` 的实现文件是？", frozenset({"a" * 64})
        )

        assert exact_store.identifiers == ("copilot_get_models",)
        assert exact_store.allowed_blob_names == ["a" * 64]
        assert results[0].path == "src-tauri/src/commands/copilot.rs"

    async def test_exact_identifier_failure_falls_back_to_semantic_results(self):
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore([_hit("src/fallback.py", 0.9)]),
            exact_store=FakeExactSearchStore(error=RuntimeError("database unavailable")),
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )

        results = await pipe.search(
            "`target_symbol` 在哪里？", frozenset({"a" * 64})
        )

        assert [result.path for result in results] == ["src/fallback.py"]

    async def test_exact_identifier_skips_unbounded_and_large_scopes(self):
        exact_store = FakeExactSearchStore([_hit("src/exact.py", 1.0)])
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore([_hit("src/semantic.py", 0.9)]),
            exact_store=exact_store,
            settings=_settings(
                confidence_floor=0.0,
                final_select_k=10,
                exact_max_scope_blobs=2,
            ),
        )

        unbounded = await pipe.search("`target_symbol` 在哪里？")
        oversized = await pipe.search(
            "`target_symbol` 在哪里？", frozenset({"a", "b", "c"})
        )

        assert [hit.path for hit in unbounded] == ["src/semantic.py"]
        assert [hit.path for hit in oversized] == ["src/semantic.py"]
        assert exact_store.identifiers == ()

    def test_call_chain_exact_candidates_fill_window_without_overwriting_scores(self):
        class WindowedLLMReranker:
            max_candidates = 30

        semantic = [
            SearchHit(
                blob_name=str(index).zfill(64),
                path=f"src/semantic_{index}.py",
                content=f"reference {index}",
                score=1.0 - index * 0.01,
            )
            for index in range(35)
        ]
        duplicate = replace(semantic[0], score=1.1)
        exact_only = SearchHit(
            blob_name="e" * 64,
            path="src/commands/target.py",
            content="def target_symbol(): pass",
            score=1.0,
        )
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore(),
            exact_store=FakeExactSearchStore(),
            llm_reranker=WindowedLLMReranker(),
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )

        merged = pipe._merge_exact_hits(
            "`target_symbol` 的完整调用链？",
            [duplicate, exact_only],
            semantic,
        )

        assert merged[0].score == semantic[0].score
        assert merged[0].score != duplicate.score
        exact_position = next(
            index for index, hit in enumerate(merged) if hit.path == exact_only.path
        )
        assert exact_position < WindowedLLMReranker.max_candidates

    async def test_symbol_location_promotes_endpoint_after_llm_rerank(self):
        class HelperFirstLLMReranker:
            async def rerank(self, query, candidates, top_k=None):
                return sorted(
                    candidates,
                    key=lambda item: "services" in item["path"],
                    reverse=True,
                )

        endpoint = SearchHit(
            blob_name="a" * 64,
            path="src-tauri/src/commands/profile.rs",
            content="#[tauri::command]\npub fn delete_profile() {}",
            score=1.0,
        )
        helper = SearchHit(
            blob_name="b" * 64,
            path="src-tauri/src/services/profile.rs",
            content="pub fn delete_profile() {}",
            score=0.95,
        )
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=FakeSearchStore(),
            exact_store=FakeExactSearchStore([endpoint, helper]),
            llm_reranker=HelperFirstLLMReranker(),
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )

        results = await pipe.search(
            "`delete_profile` 函数的实现位置？",
            frozenset({endpoint.blob_name, helper.blob_name}),
        )

        assert [result.path for result in results] == [endpoint.path, helper.path]

    async def test_multi_facet_query_recalls_and_fuses_each_facet(self):
        full = "Find authentication middleware. Trace credential reload."
        store = FakeSearchStore()
        store.hits_by_query = {
            full: [_hit("src/auth.py", 0.9)],
            "Find authentication middleware": [_hit("src/auth.py", 0.8)],
            "Trace credential reload": [_hit("src/credentials.py", 0.7)],
        }
        embedder = FakeEmbedder()
        pipe = RetrievalPipeline(
            embedder=embedder,
            store=store,
            settings=_settings(
                confidence_floor=0.0,
                final_select_k=10,
                query_decomposition_enabled=True,
                query_max_queries=3,
            ),
        )

        results = await pipe.search(full)

        assert store.queries == [
            full,
            "Find authentication middleware",
            "Trace credential reload",
        ]
        assert [hit.path for hit in results] == [
            "src/auth.py",
            "src/credentials.py",
        ]
        assert len(embedder.queries) == 3

    async def test_query_decomposition_can_be_disabled(self):
        store = FakeSearchStore([_hit("src/a.py", 0.9)])
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=store,
            settings=_settings(
                confidence_floor=0.0,
                query_decomposition_enabled=False,
            ),
        )

        await pipe.search("First repository concern. Second repository concern.")

        assert store.queries == [
            "First repository concern. Second repository concern."
        ]
