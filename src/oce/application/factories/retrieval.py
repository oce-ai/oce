"""检索栈装配工厂。

把「构造检索读路径」这件事从 composition root 里抽出来，让**冷启动**（Container）
与**热重载**（RetrievalReconfigurator）走同一条构造路径，杜绝两者漂移。

设计约束（改动前必读）：

- ``RetrievalPipeline`` 把 settings 存成 ``self.settings`` 引用，``search()`` 内部
  **调用时**读取 ``default_top_k`` / ``rrf_k`` / ``path_boost_weight`` 等；但
  ``query_planner`` / ``selector`` 是**构造期**用 settings 值固化的子对象。所以改任何
  L0 参数都必须重建 pipeline，不能只改属性。
- ``exact_max_scope_blobs`` / ``exact_timeout_seconds`` 是**双份捕获**：pipeline 调用时
  读一份，``SymbolSearchStore`` 构造时又存一份并在 ``search_exact`` 内再判一次。故本工厂
  每次都重建 ``SymbolSearchStore``（廉价对象，无 I/O）。
- ``reranker`` **不按 enabled 开关**：始终传 ``CredentialConfiguredReranker``。它的
  enabled 语义在内部 ``_resolve_config`` —— 关闭时解析出 None 再建 ``NoopReranker``。
  在 pipeline 层换成 Noop 会改变冷启动行为，热改 enabled 走 ``reload()`` 通道即可。
- ``path_store`` 按 ``path_index_enabled`` 挂/摘。注意 ``deps.path_index`` 本身也是
  flag 门控的（关闭时为 None，否则 worker 会去写路径向量）：启动时关闭、之后热开，
  拿到的仍是 None，静默无效。调用方（reconfigurator）需显式拒绝这种情况。
- LLM 三件套（llm_rerank / query_rewrite / intent）各自按 flag 决定是否构造。
  ``CredentialConfiguredLLMClient`` 构造是惰性的（首次 chat 才查库），但
  ``reload()`` 会查库，故**不能**无条件全构造 —— 那会让
  ``/admin/credentials/reload`` 多出无谓的 DB 查询。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from loguru import logger

from oce.application.queries.search import SearchQueryHandler
from oce.domain.services.retrieval import RetrievalPipeline
from oce.infrastructure.llm.credential_llm_client import CredentialConfiguredLLMClient
from oce.infrastructure.persistence.symbol_search_store import SymbolSearchStore
from oce.shared.config.settings import LLMSettings, Settings
from oce.shared.metrics import MetricsSink


@dataclass(frozen=True)
class RetrievalDeps:
    """长生命周期、持有连接的昂贵依赖。

    由 Container 构造一次并持有；热重载时**复用**，绝不重建（重建会断连接池、
    丢失凭据 reload 链路）。
    """

    embedder: object
    search_store: object
    # CredentialConfiguredReranker：始终作为 pipeline.reranker，enabled 语义在其内部
    base_reranker: object
    # PathIndexClient | None；None 表示启动时 path_index_enabled=False
    path_index: object | None
    session_factory: Callable[[], object]
    metrics: MetricsSink
    retrieval_audit_enabled: bool
    store_query_text: bool
    token_usage_cb: object | None
    llm_settings: LLMSettings


@dataclass(frozen=True)
class RetrievalStack:
    """一次构造产出的检索栈。"""

    pipeline: RetrievalPipeline
    handler: SearchQueryHandler
    symbol_store: SymbolSearchStore
    # 本次构造新建的 LLM 客户端；调用方据此同步 _CredentialRuntime 的 reload 覆盖面
    llm_clients: tuple[CredentialConfiguredLLMClient, ...]


def build_retrieval_stack(settings: Settings, deps: RetrievalDeps) -> RetrievalStack:
    """从一份完整 Settings 快照构造检索栈。

    纯构造、无副作用、可被反复调用；任何一步抛错都不会留下半生效状态（调用方在
    全部构造成功后才把它接到 query_bus 上）。

    Args:
        settings: 生效配置快照。冷启动时即 ``get_settings()``；热重载时是带 patch 的
            新快照。
        deps: 由 Container 持有的昂贵依赖，跨重载复用。

    Returns:
        可直接注册进 QueryBus 的检索栈。
    """
    retrieval = settings.retrieval
    llm = settings.llm

    symbol_store = SymbolSearchStore(
        deps.session_factory,
        max_scope_blobs=retrieval.exact_max_scope_blobs,
        timeout_seconds=retrieval.exact_timeout_seconds,
    )

    llm_clients: list[CredentialConfiguredLLMClient] = []

    llm_reranker = None
    if llm.rerank_enabled:
        rerank_llm = CredentialConfiguredLLMClient(
            "llm_rerank",
            deps.session_factory,
            llm,
            fallback_model=llm.model,
            on_usage=deps.token_usage_cb,
        )
        llm_clients.append(rerank_llm)
        from oce.domain.services.llm.reranker import LLMReranker

        llm_reranker = LLMReranker(
            client=rerank_llm,
            model=llm.model,
            max_candidates=llm.max_candidates,
            output_top_k=llm.output_top_k,
            snippet_chars=llm.snippet_chars,
        )
        logger.info("LLM reranker enabled (kind=llm_rerank)")

    query_rewriter = None
    if retrieval.query_rewrite_enabled:
        rewrite_llm = CredentialConfiguredLLMClient(
            "query_rewrite",
            deps.session_factory,
            llm,
            fallback_model=retrieval.query_rewrite_model,
            on_usage=deps.token_usage_cb,
        )
        llm_clients.append(rewrite_llm)
        from oce.domain.services.llm.rewriter import QueryRewriter

        query_rewriter = QueryRewriter(
            client=rewrite_llm,
            model=retrieval.query_rewrite_model,
            num_rewrites=retrieval.query_rewrite_num,
        )
        logger.info("Query rewriter enabled (kind=query_rewrite)")

    intent_classifier = None
    if retrieval.intent_classification_enabled:
        intent_llm = CredentialConfiguredLLMClient(
            "intent",
            deps.session_factory,
            llm,
            fallback_model=llm.model,
            on_usage=deps.token_usage_cb,
        )
        llm_clients.append(intent_llm)
        from oce.domain.services.llm.intent import IntentClassifier

        intent_classifier = IntentClassifier(
            llm_client=intent_llm,
            model=llm.model,
        )
        logger.info("Intent classifier enabled (kind=intent)")

    path_store = deps.path_index if retrieval.path_index_enabled else None

    pipeline = RetrievalPipeline(
        embedder=deps.embedder,
        store=deps.search_store,
        reranker=deps.base_reranker,
        llm_reranker=llm_reranker,
        query_rewriter=query_rewriter,
        path_store=path_store,
        exact_store=symbol_store,
        intent_classifier=intent_classifier,
        settings=retrieval,
    )
    handler = SearchQueryHandler(
        pipeline,
        metrics=deps.metrics,
        retrieval_audit_enabled=deps.retrieval_audit_enabled,
        store_query_text=deps.store_query_text,
    )
    return RetrievalStack(
        pipeline=pipeline,
        handler=handler,
        symbol_store=symbol_store,
        llm_clients=tuple(llm_clients),
    )
