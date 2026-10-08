"""意图策略在检索流程中的实际效果。

这些测试断言策略字段**真的改变行为**，而不是只存在于表里。每条都构造成
「把字段接线拆掉就会失败」：

- `split_clauses`：只有 M 才按子句多路召回，F 不拆；
- `prefer_breadth`：U 的选择结果跨文件铺开，S 允许同文件多块；
- `max_chunks_per_path`：每意图预算生效，而不是一律用全局值。
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from oce.domain.services.intent.resolver import resolve_rules
from oce.domain.services.intent.taxonomy import QueryIntent
from oce.domain.services.path_search import PathSearchResult
from oce.domain.services.retrieval import RetrievalPipeline
from oce.domain.services.retrieval_strategy import STRATEGY_TABLE
from oce.domain.services.search import SearchHit
from oce.domain.services.selector.topk_selector import TopKSelector
from oce.shared.config.settings import RetrievalSettings


def _hit(
    path: str,
    score: float,
    content: str = "code",
    *,
    start_line: int = 1,
    end_line: int | None = None,
) -> SearchHit:
    """构造命中。

    行号必须显式区分：`CoverageSelector` 按行区间抑制重叠，若同文件多块共用
    默认的 1..1，它们会被判为 100% 重叠而全部剔除，单文件预算根本轮不到生效。
    """
    return SearchHit(
        blob_name="x" * 64,
        path=path,
        content=content,
        score=score,
        start_line=start_line,
        end_line=end_line if end_line is not None else start_line + 4,
    )


def _settings(**kwargs) -> RetrievalSettings:
    return RetrievalSettings(**kwargs)


class FakeEmbedder:
    async def embed_query(self, text):
        return [float(len(text))]


class RecordingStore:
    """记录每一次 dense 检索用的查询文本，用来观察是否发生了多路召回。"""

    def __init__(self, hits: list[SearchHit]):
        self.hits = hits
        self.queries: list[str] = []

    async def search(
        self,
        *,
        query,
        query_vector,
        allowed_blob_names=None,
        top_k=50,
        vector_threshold=0.1,
    ):
        self.queries.append(query)
        return list(self.hits)


class FixedIntent:
    """固定意图的判定器替身，与 IntentResolver 同形。"""

    def __init__(self, intent: QueryIntent):
        self.intent = intent

    async def resolve(self, query: str):
        return replace(resolve_rules(query), intent=self.intent)


class RecordingPathStore:
    def __init__(self, hits: list[PathSearchResult]):
        self.hits = hits
        self.calls = 0

    async def search_paths(self, *, query_vector, allowed_blob_names=None, top_k=20):
        self.calls += 1
        return list(self.hits)


COMPOUND_QUERY = "找出 `parse_config` 的定义，并说明它的调用链路"


class TestClauseSplitIsStrategyDriven:
    """split_clauses：M 与 F 的实质差异。"""

    async def test_compound_intent_splits_into_multiple_branches(self):
        store = RecordingStore([_hit("src/a.py", 0.9)])
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=store,
            intent_classifier=FixedIntent(QueryIntent.COMPOUND),
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )
        await pipe.search(COMPOUND_QUERY)
        # 原查询 + 至少一个子句 => 多于一次 dense 召回
        assert len(store.queries) >= 2

    async def test_feature_intent_does_not_split(self):
        store = RecordingStore([_hit("src/a.py", 0.9)])
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=store,
            intent_classifier=FixedIntent(QueryIntent.FEATURE),
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )
        await pipe.search(COMPOUND_QUERY)
        # F 不拆子句：同一条可拆查询只召回一次
        assert len(store.queries) == 1

    async def test_compound_and_feature_differ_on_the_same_query(self):
        """同一查询在 M 与 F 下的召回次数必须不同，否则 M 等价于 F。"""
        counts = {}
        for intent in (QueryIntent.COMPOUND, QueryIntent.FEATURE):
            store = RecordingStore([_hit("src/a.py", 0.9)])
            pipe = RetrievalPipeline(
                embedder=FakeEmbedder(),
                store=store,
                intent_classifier=FixedIntent(intent),
                settings=_settings(confidence_floor=0.0, final_select_k=10),
            )
            await pipe.search(COMPOUND_QUERY)
            counts[intent] = len(store.queries)
        assert counts[QueryIntent.COMPOUND] > counts[QueryIntent.FEATURE]

    async def test_split_still_happens_without_intent_classification(self):
        """未启用意图分类时退回检测器驱动，不能连带丢掉复合召回。"""
        store = RecordingStore([_hit("src/a.py", 0.9)])
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=store,
            intent_classifier=None,
            settings=_settings(confidence_floor=0.0, final_select_k=10),
        )
        await pipe.search(COMPOUND_QUERY)
        assert len(store.queries) >= 2


class TestBreadthAndBudget:
    """prefer_breadth 与 max_chunks_per_path：U 与 S 的实质差异。"""

    #: 同一文件 4 块 + 另一文件 1 块，用来观察单文件是否垄断预算。
    HITS = [
        _hit("src/caller.py", 0.95, "call one", start_line=10),
        _hit("src/caller.py", 0.94, "call two", start_line=100),
        _hit("src/caller.py", 0.93, "call three", start_line=200),
        _hit("src/caller.py", 0.92, "call four", start_line=300),
        _hit("src/other.py", 0.50, "call five", start_line=10),
    ]

    async def _run(self, intent: QueryIntent) -> list[SearchHit]:
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=RecordingStore(list(self.HITS)),
            intent_classifier=FixedIntent(intent),
            settings=_settings(
                confidence_floor=0.0,
                final_select_k=10,
                max_chunks_per_path=3,
                overlap_threshold=1.0,  # 仅完全相同才抑制；各块内容不同
            ),
        )
        return await pipe.search("q")

    async def test_usage_intent_spreads_across_files(self):
        """U 的单文件上限压到 1，引用点必须跨文件铺开。"""
        results = await self._run(QueryIntent.USAGE)
        per_file = {}
        for hit in results:
            per_file[hit.path] = per_file.get(hit.path, 0) + 1
        assert per_file.get("src/caller.py") == 1
        assert "src/other.py" in per_file

    async def test_symbol_intent_allows_multiple_chunks_per_file(self):
        """S 不广度优先，允许同文件多块（策略预算为 2）。"""
        results = await self._run(QueryIntent.SYMBOL)
        same_file = [h for h in results if h.path == "src/caller.py"]
        assert len(same_file) == STRATEGY_TABLE[QueryIntent.SYMBOL].max_chunks_per_path
        assert len(same_file) > 1

    async def test_usage_and_symbol_differ_on_the_same_hits(self):
        """同一候选集在 U 与 S 下的单文件块数必须不同。"""
        usage = await self._run(QueryIntent.USAGE)
        symbol = await self._run(QueryIntent.SYMBOL)

        def same_file_count(hits):
            return len([h for h in hits if h.path == "src/caller.py"])

        assert same_file_count(usage) < same_file_count(symbol)

    async def test_per_intent_budget_overrides_global_setting(self):
        """策略预算必须压过全局 max_chunks_per_path。

        全局设 5、策略（CALL_CHAIN）为 3；若选择器仍用全局值就会拿到 4 块。
        """
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=RecordingStore(list(self.HITS)),
            intent_classifier=FixedIntent(QueryIntent.CALL_CHAIN),
            settings=_settings(
                confidence_floor=0.0,
                final_select_k=10,
                max_chunks_per_path=5,
                overlap_threshold=1.0,
            ),
        )
        results = await pipe.search("q")
        same_file = [h for h in results if h.path == "src/caller.py"]
        expected = STRATEGY_TABLE[QueryIntent.CALL_CHAIN].max_chunks_per_path
        assert len(same_file) == expected == 3

    @pytest.mark.parametrize("has_path_hits", [False, True])
    async def test_path_branch_uses_per_intent_budget(self, has_path_hits):
        hits = list(self.HITS[:4])
        path_hits = [PathSearchResult(hits[0].path, hits[0].blob_name, 0.9)]
        path_store = RecordingPathStore(path_hits if has_path_hits else [])
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=RecordingStore(hits),
            path_store=path_store,
            intent_classifier=FixedIntent(QueryIntent.PATH),
            settings=_settings(
                confidence_floor=0.0,
                final_select_k=10,
                max_chunks_per_path=5,
            ),
        )

        results = await pipe.search("src/caller.py")

        assert path_store.calls > 0
        assert len(results) == STRATEGY_TABLE[QueryIntent.PATH].max_chunks_per_path == 2

    async def test_path_branch_without_intent_uses_global_budget(self):
        hits = list(self.HITS[:4])
        path_store = RecordingPathStore(
            [PathSearchResult(hits[0].path, hits[0].blob_name, 0.9)]
        )
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=RecordingStore(hits),
            path_store=path_store,
            settings=_settings(
                confidence_floor=0.0,
                final_select_k=10,
                max_chunks_per_path=4,
            ),
        )

        results = await pipe.search("src/caller.py")

        assert path_store.calls > 0
        assert len(results) == 4

    @pytest.mark.parametrize("intent", [QueryIntent.PATH, QueryIntent.FEATURE])
    async def test_custom_selector_is_preserved_for_every_branch(self, intent):
        hits = list(self.HITS[:4])
        path_store = RecordingPathStore([])
        pipe = RetrievalPipeline(
            embedder=FakeEmbedder(),
            store=RecordingStore(hits),
            path_store=path_store,
            selector=TopKSelector(),
            intent_classifier=FixedIntent(intent),
            settings=_settings(
                confidence_floor=0.0,
                final_select_k=10,
                max_chunks_per_path=2,
            ),
        )
        query = "src/caller.py" if intent is QueryIntent.PATH else "retry behavior"

        results = await pipe.search(query)

        assert bool(path_store.calls) is (intent is QueryIntent.PATH)
        assert len(results) == 4


class TestEveryStrategyFieldIsConsumed:
    """守卫：策略字段不得是占位。"""

    def test_declared_fields_are_referenced_by_the_pipeline(self):
        """每个策略字段都必须在 src/ 下被读取，否则它是死配置。

        豁免集里的四个字段都是本次重构之前就存在的未接线开关：
        `enable_multi_hop` / `enable_reference_graph` 在表中恒为 False；
        `boost_definitions` / `boost_docs` 有非默认取值但检索层从未读取，
        属于既有技术债，不在本次范围内。其余字段必须有真实消费点。
        """
        import pathlib
        import re

        from oce.domain.services.retrieval_strategy import RetrievalStrategy

        root = pathlib.Path(__file__).resolve().parents[3] / "src"
        sources = "\n".join(
            p.read_text(encoding="utf-8")
            for p in root.rglob("*.py")
            if "__pycache__" not in p.parts and p.name != "retrieval_strategy.py"
        )
        exempt = {
            "enable_multi_hop",
            "enable_reference_graph",
            "boost_definitions",
            "boost_docs",
        }
        unconsumed = [
            name
            for name in RetrievalStrategy.__dataclass_fields__
            if name not in exempt
            and not re.search(rf"\.{re.escape(name)}\b", sources)
        ]
        assert not unconsumed, f"策略字段已声明但无人消费（占位配置）: {unconsumed}"
