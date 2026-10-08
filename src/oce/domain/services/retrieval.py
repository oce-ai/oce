"""RetrievalPipeline 领域服务 - 检索编排

流程（与旧 pipeline 语义对齐）：
    embed_query → store.search（dense 向量检索）
    → 精确标识符召回 → 多查询结果融合 → rerank（逐篇精排）
    → 源码优先（降权文档/测试）→ 置信度门槛 → select（最终 K 条）

rerank 解决「单篇多相关」，select 解决「这一组够全且不冗余」，职责不同。
关闭查询分解、使用 Noop reranker 和自定义 TopK selector 时，可退化为传统 Top-K。
"""

from __future__ import annotations

import asyncio
import re
from contextlib import contextmanager
from dataclasses import replace
from typing import Callable, Iterator

from loguru import logger

from oce.domain.services.embedder import Embedder
from oce.domain.services.intent.resolver import IntentResolver, resolve_rules
from oce.domain.services.intent.signals import extract_symbols, split_independent_clauses
from oce.domain.services.intent.taxonomy import QueryIntent
from oce.domain.services.path_search import PathSearchStore
from oce.domain.services.query_classifier import should_use_path_index
from oce.domain.services.query_planner import HeuristicQueryPlanner, QueryPlanner
from oce.domain.services.reranker import NoopReranker, Reranker
from oce.domain.services.retrieval_strategy import RetrievalStrategy, get_strategy
from oce.domain.services.search import ExactSearchStore, SearchHit, SearchStore, search_hit_key
from oce.domain.services.selector.coverage_selector import CoverageSelector
from oce.domain.services.selector.protocols import Selector
from oce.shared.config import get_settings
from oce.shared.config.settings import RetrievalSettings
from oce.shared.metrics import RetrievalAudit

# LLM reranker (optional)
try:
    from oce.domain.services.llm.reranker import LLMReranker
except ImportError:
    LLMReranker = None  # type: ignore


@contextmanager
def _noop_stage(_name: str) -> Iterator[None]:
    """audit=None 时的空计时上下文：不测量、零副作用。"""
    yield


def source_priority_factor(path: str) -> float:
    """源码优先先验：文档/测试类路径乘性降权，普通源码 1.0。

    与旧 retrieval/ranking.py 的规则集对齐（简化版，可注入自定义函数）。
    """
    p = path.replace("\\", "/").lower()
    name = p.rsplit("/", 1)[-1]
    stem = name.split(".", 1)[0]

    # 法律文件最重降权
    if stem in {"license", "notice", "copying"}:
        return 0.1
    # 主 README 显式不降权，多语言 README 降权
    if stem == "readme":
        return 1.0
    if stem.startswith("readme"):
        return 0.2
    # 文档目录 / 文档扩展
    if "/docs/" in f"/{p}" or p.endswith((".md", ".rst", ".txt")):
        return 0.5
    if (
        "/tests/" in f"/{p}"
        or name.startswith("test_")
        or name == "conftest.py"
        or ".test." in name
        or ".spec." in name
    ):
        return 0.6
    if name in {"index.ts", "index.tsx", "index.js", "index.jsx", "types.ts"}:
        return 0.85
    return 1.0


def path_query_priority_factor(path: str) -> float:
    """路径类查询的优先级因子：恒为 1.0（文档中立）。

    「XX 文件在哪里」类查询中，任何文件类型都可能是目标（文档、配置、测试、
    license 都在找文件语义内），source_priority_factor 的「源码优先」先验不成立。
    路径类查询不乘性降权，排序完全交由路径 boost + 内容分数 + rerank 决定。
    """
    return 1.0


class RetrievalPipeline:
    """检索管道：编排检索全流程"""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: SearchStore,
        reranker: Reranker | None = None,
        llm_reranker: LLMReranker | None = None,
        query_rewriter: "QueryRewriter | None" = None,
        path_store: PathSearchStore | None = None,  # 路径索引
        exact_store: ExactSearchStore | None = None,
        selector: Selector | None = None,
        query_planner: QueryPlanner | None = None,
        priority_factor: Callable[[str], float] | None = None,
        settings: RetrievalSettings | None = None,
        intent_classifier: IntentResolver | None = None,  # 意图判定器
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.reranker = reranker or NoopReranker()
        self.llm_reranker = llm_reranker  # Optional LLM-based semantic reranker
        self.query_rewriter = query_rewriter  # Optional query rewriter for better recall
        self.path_store = path_store  # Optional path index for filename queries
        self.exact_store = exact_store
        self.priority_factor = priority_factor or source_priority_factor
        self.settings = settings or get_settings().retrieval
        self.intent_classifier = intent_classifier  # Optional intent classifier
        self.query_planner = query_planner or HeuristicQueryPlanner(
            max_queries=(
                self.settings.query_max_queries
                if self.settings.query_decomposition_enabled
                else 1
            ),
            min_facet_chars=self.settings.query_min_facet_chars,
        )
        self._has_custom_selector = selector is not None
        self.selector = (
            selector
            if selector is not None
            else CoverageSelector(
                max_per_path=self.settings.max_chunks_per_path,
                max_chars=self.settings.max_context_chars,
                overlap_threshold=self.settings.overlap_threshold,
            )
        )

    async def search(
        self,
        query: str,
        allowed_blob_names: frozenset[str] | None = None,
        *,
        audit: RetrievalAudit | None = None,
    ) -> list[SearchHit]:
        """执行一次检索，返回最终命中列表（按融合分降序）。

        Args:
            query: 查询文本
            allowed_blob_names: 允许搜索的 blob 名称集合
                - None: 搜索所有 blobs（不过滤）
                - 空集合: 没有可搜索的 blobs（返回空结果）
            audit: 可选的阶段耗时收集容器；为 None 时不打点、零开销。
        """
        stage = audit.stage if audit is not None else _noop_stage
        if audit is not None:
            audit.scope_size = (
                len(allowed_blob_names) if allowed_blob_names is not None else None
            )

        # None 表示不过滤，空集合表示无可搜索内容
        if allowed_blob_names is not None and len(allowed_blob_names) == 0:
            return []

        # 意图驱动的策略选择
        strategy = None
        detected_intent = None
        intent_decision = None
        if self.intent_classifier is not None:
            try:
                with stage("intent"):
                    intent_decision = await self.intent_classifier.resolve(query)
                    detected_intent = intent_decision.intent
            except Exception as e:
                # 判定器内部已对 provider 故障做降级，这里只兜住真正的意外
                # 异常；纯规则判定为本地正则，恒定可用。
                logger.warning(
                    f"Intent resolution failed: {e}, falling back to rules-only"
                )
                intent_decision = resolve_rules(query)
                detected_intent = intent_decision.intent
            strategy = get_strategy(detected_intent)
            logger.debug(f"Query intent: {detected_intent.value}, strategy: {strategy}")
        if audit is not None and detected_intent is not None:
            audit.intent = detected_intent.value
            if intent_decision is not None:
                audit.intent_source = intent_decision.source
                audit.intent_decision_reason = intent_decision.reason

        # 路径索引增强：根据意图或启发式判断
        use_path_index = (
            (strategy and strategy.enable_path_index)
            or (self.path_store and should_use_path_index(query))
        )
        if use_path_index and self.path_store:
            if audit is not None:
                audit.path_boosted = True
            return await self._search_with_path_boost(
                query,
                allowed_blob_names,
                enable_query_rewrite=(
                    strategy.enable_query_rewrite
                    if strategy is not None
                    else self.query_rewriter is not None
                ),
                enable_llm_rerank=(
                    strategy.enable_llm_rerank
                    if strategy is not None
                    else self.llm_reranker is not None
                ),
                selector=self._selector_for(strategy),
                audit=audit,
            )

        # Query rewrite: 根据意图决定是否启用
        queries_to_search = [query]
        use_query_rewrite = (
            (strategy and strategy.enable_query_rewrite)
            or (strategy is None and self.query_rewriter is not None)
        )
        if use_query_rewrite and self.query_rewriter is not None:
            with stage("rewrite"):
                rewritten_queries = await self.query_rewriter.rewrite(query)
            # 使用改写的查询替换原查询
            if rewritten_queries:
                queries_to_search = rewritten_queries

        # 复合查询拆分：M 意图的实质差异。策略表里只有 COMPOUND 的
        # ``split_clauses`` 为真，这正是 M 区别于 F 的地方。
        # 未启用意图分类时（strategy is None）退回检测器驱动，保持旧行为，
        # 否则关掉意图分类会连带丢掉复合召回。
        use_clause_split = (
            strategy.split_clauses if strategy is not None else True
        )
        compound_parts = (
            split_independent_clauses(query) if use_clause_split else [query]
        )
        if len(compound_parts) >= 2:
            known = set(queries_to_search)
            queries_to_search.extend(
                part for part in compound_parts if part not in known
            )
            if audit is not None:
                audit.compound_parts = compound_parts
            logger.info(
                "Compound query split into %d retrieval branches",
                len(queries_to_search),
            )

        # 对每个查询执行完整检索流程
        all_result_lists = []
        with stage("dense"):
            for search_query in queries_to_search:
                planned_queries = self.query_planner.plan(search_query)
                if not planned_queries:
                    continue
                num_queries = len(planned_queries)
                result_lists = await asyncio.gather(
                    *(
                        self._recall(planned_query, allowed_blob_names, num_queries)
                        for planned_query in planned_queries
                    )
                )
                all_result_lists.extend(result_lists)

        with stage("exact"):
            exact_hits = await self._recall_exact(query, allowed_blob_names)
        if not all_result_lists and not exact_hits:
            return []

        with stage("fuse"):
            hits = self._fuse(all_result_lists) if all_result_lists else []
            hits = self._merge_exact_hits(query, exact_hits, hits, intent=detected_intent)
        with stage("rerank"):
            hits = await self.reranker.rerank(query, hits)
        hits = self._apply_source_priority(hits)

        # LLM-based semantic rerank: 根据意图决定是否启用
        use_llm_rerank = (
            (strategy and strategy.enable_llm_rerank)
            or (strategy is None and self.llm_reranker is not None)
        )
        if use_llm_rerank:
            with stage("llm_rerank"):
                hits = await self._llm_rerank_hits(query, hits)
        hits = self._promote_symbol_endpoints(query, hits, intent=detected_intent)

        with stage("select"):
            hits = self._apply_confidence_floor(hits)
            selector = self._selector_for(strategy)
            selected = await selector.select(hits, self.settings.final_select_k)
        return selected

    def _selector_for(self, strategy: RetrievalStrategy | None) -> Selector:
        """按意图返回选择器；预算与默认一致时复用共享实例。

        显式注入的选择器拥有选择策略；意图预算只调整默认 CoverageSelector。
        默认选择器在预算不同时才新建，避免每次请求都分配对象。

        ``prefer_breadth`` 是 U（引用点）的实质差异：引用点要跨文件铺开，
        单文件上限压到 1，防止一个文件吃掉整个预算。
        """
        if strategy is None or self._has_custom_selector:
            return self.selector

        max_per_path = strategy.max_chunks_per_path
        if strategy.prefer_breadth:
            max_per_path = 1
        if max_per_path == self.settings.max_chunks_per_path:
            return self.selector
        return CoverageSelector(
            max_per_path=max_per_path,
            max_chars=self.settings.max_context_chars,
            overlap_threshold=self.settings.overlap_threshold,
        )

    async def _llm_rerank_hits(
        self, query: str, hits: list[SearchHit]
    ) -> list[SearchHit]:
        """用 LLM 语义重排候选。

        必须带上 content 与行号：只给路径会让重排退化成文件名匹配，符号定义
        和调用链查询无从判断。search() 与 _search_with_path_index() 共用此
        方法，避免两处候选字段再次分叉。
        """
        if self.llm_reranker is None:
            return hits

        candidates = [
            {
                "path": hit.path,
                "score": hit.score,
                "content": hit.content,
                "start_line": hit.start_line,
                "end_line": hit.end_line,
                "hit": hit,
            }
            for hit in hits
        ]
        reranked = await self.llm_reranker.rerank(query, candidates)
        return [c["hit"] for c in reranked if "hit" in c]

    async def _recall(
        self,
        query: str,
        allowed_blob_names: set[str] | None,
        num_queries: int = 1,
    ) -> list[SearchHit]:
        # 动态调整召回量：单查询用 default_top_k，多查询用 per_query_top_k
        top_k = (
            self.settings.default_top_k
            if num_queries == 1
            else self.settings.per_query_top_k
        )

        query_vector = await self.embedder.embed_query(query)

        return await self.store.search(
            query=query,
            query_vector=query_vector,
            allowed_blob_names=(
                sorted(allowed_blob_names) if allowed_blob_names is not None else None
            ),
            top_k=top_k,
            vector_threshold=self.settings.vector_threshold,
        )

    async def _recall_exact(
        self,
        query: str,
        allowed_blob_names: frozenset[str] | None,
    ) -> list[SearchHit]:
        if self.exact_store is None:
            return []
        scope_limit = self.settings.exact_max_scope_blobs
        if (
            allowed_blob_names is None
            or scope_limit == 0
            or len(allowed_blob_names) > scope_limit
        ):
            logger.debug(
                "Skipping SQL exact recall for scope size {} (limit {})",
                len(allowed_blob_names) if allowed_blob_names is not None else "unbounded",
                scope_limit,
            )
            return []
        identifiers = extract_symbols(query)
        if not identifiers:
            return []
        try:
            return await self.exact_store.search_exact(
                identifiers=identifiers,
                allowed_blob_names=(
                    sorted(allowed_blob_names) if allowed_blob_names is not None else None
                ),
                top_k=self.settings.default_top_k,
            )
        except Exception as exc:
            logger.warning("Exact identifier recall failed; using semantic candidates: {}", exc)
            return []

    def _merge_exact_hits(
        self,
        query: str,
        exact_hits: list[SearchHit],
        semantic_hits: list[SearchHit],
        *,
        intent: QueryIntent | None = None,
    ) -> list[SearchHit]:
        effective_intent = intent or resolve_rules(query).intent
        if (
            effective_intent == QueryIntent.CALL_CHAIN
            and semantic_hits
        ):
            semantic_keys = {search_hit_key(hit) for hit in semantic_hits}
            exact_only = [
                hit for hit in exact_hits if search_hit_key(hit) not in semantic_keys
            ]
            candidate_window = min(
                self.settings.default_top_k,
                getattr(
                    self.llm_reranker,
                    "max_candidates",
                    self.settings.default_top_k,
                ),
            )
            reserved = min(len(exact_only), max(1, candidate_window // 3))
            semantic_slots = max(candidate_window - reserved, 1)
            anchor_index = min(semantic_slots, len(semantic_hits)) - 1
            anchor_score = semantic_hits[anchor_index].score
            exact_only = [
                replace(hit, score=min(hit.score, anchor_score))
                for hit in exact_only[:reserved]
            ]
            merged = [*semantic_hits, *exact_only]
            merged.sort(key=lambda hit: hit.score, reverse=True)
            return merged[: self.settings.default_top_k]

        merged: list[SearchHit] = []
        positions: dict[tuple[str, str, int, int, str], int] = {}
        for hit in [*exact_hits, *semantic_hits]:
            key = search_hit_key(hit)
            position = positions.get(key)
            if position is None:
                positions[key] = len(merged)
                merged.append(hit)
            elif hit.score > merged[position].score:
                merged[position] = replace(hit, score=hit.score)
        merged.sort(key=lambda hit: hit.score, reverse=True)
        return merged[: self.settings.default_top_k]

    @staticmethod
    def _promote_symbol_endpoints(
        query: str,
        hits: list[SearchHit],
        *,
        intent: QueryIntent | None = None,
    ) -> list[SearchHit]:
        """符号定位时，框架 endpoint 定义稳定优先于同名内部实现。"""
        effective_intent = intent or resolve_rules(query).intent
        if effective_intent != QueryIntent.SYMBOL:
            return hits
        location_markers = (
            "实现位置",
            "实现文件",
            "源码位置",
            "哪个文件",
            "在哪个文件",
            "在哪里定义",
            "哪里定义",
            "函数在哪",
            "defined",
            "definition",
            "implementation",
        )
        if not any(marker in query.casefold() for marker in location_markers):
            return hits
        identifiers = extract_symbols(query)
        if not identifiers:
            return hits

        endpoint_patterns = [
            re.compile(
                rf"(?ms)(?:#\[(?:tauri::command|pytauri::command)[^]]*\]|"
                rf"@(?:app|router)\.(?:get|post|put|patch|delete)\([^\n]*\))"
                rf"\s*(?:(?:pub|export)(?:\([^)]*\))?\s+)?"
                rf"(?:(?:async|default)\s+)?(?:fn|def|function)\s+"
                rf"{re.escape(identifier.rsplit('::', 1)[-1])}\b"
            )
            for identifier in identifiers
        ]
        return sorted(
            hits,
            key=lambda hit: any(pattern.search(hit.content) for pattern in endpoint_patterns),
            reverse=True,
        )

    def _fuse(self, result_lists: list[list[SearchHit]]) -> list[SearchHit]:
        if len(result_lists) == 1:
            return result_lists[0]

        rrf_k = self.settings.rrf_k  # 统一使用 rrf_k
        weights = [1.0] + [self.settings.query_facet_weight] * (len(result_lists) - 1)
        max_score = sum(weight / (rrf_k + 1) for weight in weights)
        scores: dict[tuple[str, str, int, int, str], float] = {}
        hits_by_key: dict[tuple[str, str, int, int, str], SearchHit] = {}
        first_seen: dict[tuple[str, str, int, int, str], int] = {}
        ordinal = 0
        for weight, hits in zip(weights, result_lists):
            for rank, hit in enumerate(hits, 1):
                key = search_hit_key(hit)
                if key not in first_seen:
                    first_seen[key] = ordinal
                    ordinal += 1
                    hits_by_key[key] = hit
                scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)

        keys = sorted(scores, key=lambda key: (-scores[key], first_seen[key]))
        return [
            replace(hits_by_key[key], score=scores[key] / max_score)
            for key in keys[: self.settings.default_top_k]
        ]

    def _apply_source_priority(
        self,
        hits: list[SearchHit],
        priority_factor: Callable[[str], float] | None = None,
    ) -> list[SearchHit]:
        """按 score × 路径惩罚因子稳定重排（只重排，不改 rerank 决策）。

        普通查询默认用 self.priority_factor（源码优先）；路径类查询可传
        文档中立的 factor（path_query_priority_factor）。
        """
        factor = priority_factor or self.priority_factor
        if not hits:
            return hits
        return sorted(
            hits,
            key=lambda h: h.score * factor(h.path),
            reverse=True,
        )

    def _apply_confidence_floor(
        self,
        hits: list[SearchHit],
        priority_factor: Callable[[str], float] | None = None,
    ) -> list[SearchHit]:
        """逐条按有效分（score × penalty）剔除低于门槛的弱匹配"""
        factor = priority_factor or self.priority_factor
        floor = self.settings.confidence_floor
        return [h for h in hits if h.score * factor(h.path) >= floor]


    async def _search_with_path_boost(
        self,
        query: str,
        allowed_blob_names: frozenset[str] | None = None,
        *,
        enable_query_rewrite: bool,
        enable_llm_rerank: bool,
        selector: Selector,
        audit: RetrievalAudit | None = None,
    ) -> list[SearchHit]:
        """
        使用路径索引增强的检索（用于文件名查询）

        路径索引回答「哪个文件」，内容索引回答「文件里哪一段」，两者按 chunk 粒度
        合并：路径命中的文件若已有内容命中则加权提分，不替换，否则会把真正含目标
        符号的 chunk 挤掉（回填的文件首个 chunk 通常只是 use / import 语句）。
        """
        logger.info(f"Path-boosted search for query: {query}")
        stage = audit.stage if audit is not None else _noop_stage

        # 0. 查询改写变体（路径索引与内容索引共用，解决中文查询 vs 英文文件名）
        queries_to_search = [query]
        if enable_query_rewrite and self.query_rewriter is not None:
            with stage("rewrite"):
                try:
                    rewritten_queries = await self.query_rewriter.rewrite(query)
                    if rewritten_queries:
                        queries_to_search = rewritten_queries
                except Exception as e:
                    logger.warning(f"Query rewrite failed: {e}")

        # 1. 路径索引检索：原查询 + 改写变体分别检索，每个 blob 取最高路径分。
        #    中文查询（如「版本变更历史记录文件在哪里」）直接 embedding 常匹配不到
        #    英文路径文档，改写变体（含文件名如 CHANGES.rst）才能命中路径索引。
        path_scores: dict[str, float] = {}
        with stage("dense"):
            try:
                blob_filter = list(allowed_blob_names) if allowed_blob_names else None
                for variant in [query, *queries_to_search]:
                    query_vector = await self.embedder.embed_query(variant)
                    path_results = await self.path_store.search_paths(
                        query_vector=query_vector,
                        allowed_blob_names=blob_filter,
                        top_k=20,
                    )
                    for r in path_results:
                        if r.blob_name not in path_scores or r.score > path_scores[r.blob_name]:
                            path_scores[r.blob_name] = r.score
                logger.info(f"Path index returned {len(path_scores)} results")
            except Exception as e:
                logger.warning(f"Path index search failed: {e}, falling back to content-only")

        # 2. 内容索引检索（常规流程，但减少 top_k）
        content_hits = []
        with stage("dense"):
            try:
                all_result_lists = []
                for search_query in queries_to_search:
                    planned_queries = self.query_planner.plan(search_query)
                    if not planned_queries:
                        continue
                    num_queries = len(planned_queries)
                    result_lists = await asyncio.gather(
                        *(
                            self._recall(planned_query, allowed_blob_names, num_queries)
                            for planned_query in planned_queries
                        )
                    )
                    all_result_lists.extend(result_lists)

                if all_result_lists:
                    content_hits = self._fuse(all_result_lists)
            except Exception as e:
                logger.warning(f"Content search failed: {e}")

        # 3. 融合：路径分数作为文件级加权，排序仍在 chunk 粒度上进行
        with stage("fuse"):
            if not path_scores:
                # 路径索引失败，回退到纯内容检索
                logger.info("No path results, using content-only")
                hits = content_hits
            else:
                hits = await self._merge_path_and_content(path_scores, content_hits)

        # 4. 应用常规后处理
        with stage("rerank"):
            hits = await self.reranker.rerank(query, hits)
        # 路径类查询使用文档中立的优先级因子（不降权 .rst/.md/.txt）
        # 避免「版本变更历史文件在哪里」被 .rst 文档降权压出 Top-10
        hits = self._apply_source_priority(hits, priority_factor=path_query_priority_factor)

        if enable_llm_rerank:
            with stage("llm_rerank"):
                hits = await self._llm_rerank_hits(query, hits)

        with stage("select"):
            hits = self._apply_confidence_floor(hits, priority_factor=path_query_priority_factor)
            selected = await selector.select(hits, self.settings.final_select_k)
        return selected

    async def _merge_path_and_content(
        self,
        path_scores: dict[str, float],
        content_hits: list[SearchHit],
    ) -> list[SearchHit]:
        """按 chunk 粒度合并路径命中与内容命中。

        旧实现按 path 去重并让路径结果优先，导致路径索引越准、正确 chunk 越会被
        丢弃：「`add_provider` 在哪个文件定义？」路径索引选对 provider.rs 后，
        内容索引里含该函数的 chunk 因同属该文件而被整块剔除。
        """
        weight = self.settings.path_boost_weight

        merged: list[SearchHit] = []
        covered: set[str] = set()
        for hit in content_hits:
            boost = path_scores.get(hit.blob_name)
            if boost is None:
                merged.append(hit)
                continue
            covered.add(hit.blob_name)
            merged.append(replace(hit, score=hit.score + weight * boost))

        # 内容检索完全没覆盖到的文件才回填首个 chunk，保住纯文件名查询的召回
        missing = [name for name in path_scores if name not in covered]
        if missing:
            merged.extend(await self._fetch_content_for_paths(missing, path_scores))

        logger.info(
            f"Merged {len(content_hits)} content hits with {len(path_scores)} path hits "
            f"(boosted={len(covered)}, backfilled={len(missing)})"
        )
        merged.sort(key=lambda hit: hit.score, reverse=True)
        return merged

    async def _fetch_content_for_paths(
        self,
        blob_names: list[str],
        path_scores: dict[str, float],
    ) -> list[SearchHit]:
        """
        为路径索引结果获取实际内容

        仅用于内容索引完全没召回的文件：取首个 chunk 作为代表，让纯文件名查询
        至少能命中目标文件。
        """
        hits = []

        # 使用一个虚拟查询来获取这些 blob 的内容
        # 这里我们需要直接访问数据库，因为 SearchStore 不提供按 blob_name 查询的接口
        from oce.infrastructure.persistence.models import BlobChunkModel, BlobModel, ChunkModel
        from oce.shared.database.session import get_session_factory
        from sqlalchemy import select

        try:
            async with get_session_factory()() as session:
                # 为每个 blob 获取第一个 chunk
                for blob_name in blob_names:
                    stmt = (
                        select(
                            ChunkModel.content_hash,
                            ChunkModel.content,
                            BlobChunkModel.start_line,
                            BlobChunkModel.end_line,
                            BlobModel.blob_name,
                            BlobModel.path,
                        )
                        .join(BlobChunkModel, BlobChunkModel.content_hash == ChunkModel.content_hash)
                        .join(BlobModel, BlobModel.blob_name == BlobChunkModel.blob_name)
                        .where(BlobModel.blob_name == blob_name)
                        .order_by(BlobChunkModel.start_line)
                        .limit(1)
                    )

                    result = await session.execute(stmt)
                    row = result.first()

                    if row:
                        hits.append(
                            SearchHit(
                                content_hash=row.content_hash,
                                blob_name=row.blob_name,
                                path=row.path,
                                start_line=row.start_line,
                                end_line=row.end_line,
                                content=row.content,
                                score=path_scores.get(blob_name, 0.9),  # 使用路径索引的分数
                            )
                        )
        except Exception as e:
            logger.error(f"Failed to fetch content for paths: {e}")

        return hits
