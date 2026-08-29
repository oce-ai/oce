"""进程级 composition root。"""

from __future__ import annotations

import os
from functools import lru_cache

from loguru import logger

from oce.application.bus import CommandBus, QueryBus
from oce.application.commands.checkpoint import CheckpointCommand, CheckpointCommandHandler
from oce.application.commands.credentials import (
    ReloadEmbeddingCredentialsCommand,
    ReloadEmbeddingCredentialsCommandHandler,
)
from oce.application.commands.gc import GcCommand, GcCommandHandler
from oce.application.commands.ingest import (
    DeleteBlobsCommand,
    DeleteBlobsCommandHandler,
    EmbedPendingCommand,
    EmbedPendingCommandHandler,
    IngestBlobCommand,
    IngestBlobCommandHandler,
    IngestBlobsCommand,
    IngestBlobsCommandHandler,
)
from oce.application.commands.queue_admin import (
    ResetQueueCommand,
    ResetQueueCommandHandler,
)
from oce.application.commands.requeue import (
    RequeueStaleCommand,
    RequeueStaleCommandHandler,
)
from oce.application.credential_admin import (
    CreateCredentialCommand,
    CreateCredentialCommandHandler,
    DeleteCredentialCommand,
    DeleteCredentialCommandHandler,
    DuplicateCredentialCommand,
    DuplicateCredentialCommandHandler,
    ListCredentialsQuery,
    ListCredentialsQueryHandler,
    UpdateCredentialCommand,
    UpdateCredentialCommandHandler,
)
from oce.application.queries.queue import (
    QueueStatusQuery,
    QueueStatusQueryHandler,
)
from oce.application.queries.search import SearchQuery, SearchQueryHandler
from oce.application.queries.stats import (
    MonitoringStatsQuery,
    MonitoringStatsQueryHandler,
)
from oce.application.queries.status import (
    BlobStatusQuery,
    BlobStatusQueryHandler,
    FindMissingQuery,
    FindMissingQueryHandler,
    ResolveScopeQuery,
    ResolveScopeQueryHandler,
)
from oce.application.service import RetrievalApplication
from oce.application.factories.chunker import build_chunker
from oce.application.worker import EmbedWorker
from oce.domain.services.retrieval import RetrievalPipeline
from oce.infrastructure.embed.credential_embedder import CredentialConfiguredEmbedder
from oce.infrastructure.embed.credential_reranker import CredentialConfiguredReranker
from oce.infrastructure.llm.credential_llm_client import CredentialConfiguredLLMClient
from oce.infrastructure.milvus3 import Milvus3SearchStore
from oce.infrastructure.milvus3.path_index import PathIndexClient
from oce.infrastructure.persistence.credential_admin_store import (
    SqlCredentialAdminStore,
)
from oce.infrastructure.persistence.symbol_search_store import SymbolSearchStore
from oce.infrastructure.persistence.uow import SqlAlchemyUnitOfWork
from oce.infrastructure.metrics.cleanup import MonitoringCleaner
from oce.infrastructure.metrics.resource_sampler import (
    ResourceSampler,
    build_psutil_collector,
)
from oce.infrastructure.metrics.sql_metrics_sink import SqlMetricsSink
from oce.infrastructure.metrics.stats_store import SqlMonitoringStatsReader
from oce.infrastructure.queue.redis_queue import RedisQueue
from oce.shared.config import get_settings
from oce.shared.database.session import async_session_factory
from oce.shared.logging import DATA_DIR_ENV
from oce.shared.metrics import NoopMetricsSink, TokenUsageRecord


class _CredentialRuntime:
    def __init__(self, embedder, reranker, llm_clients=()) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self._llm_clients = [client for client in llm_clients if client is not None]

    async def reload(self) -> int:
        embedding_replacement = await self._embedder.prepare_reload()
        try:
            rerank_replacement = await self._reranker.prepare_reload()
        except Exception:
            await self._embedder.discard_prepared(embedding_replacement)
            raise
        try:
            pool_size = await self._embedder.activate_prepared(embedding_replacement)
        except Exception:
            await self._reranker.discard_prepared(rerank_replacement)
            raise
        await self._reranker.activate_prepared(rerank_replacement)
        # LLM 客户端无预备/激活两阶段（reload 仅原子替换 delegate）；旁路容错，
        # 单个刷新失败不回滚已激活的 embedder/reranker，只记日志。
        for client in self._llm_clients:
            try:
                await client.reload()
            except Exception as exc:
                logger.warning("LLM client reload failed: {}", exc)
        return pool_size


class Container:
    def __init__(self) -> None:
        settings = get_settings()
        if settings.embedding.dimensions != settings.milvus.dense_dim:
            raise ValueError("EMBED_DIMENSIONS must equal MILVUS_DENSE_DIM")

        # token 用量回调：监控开启时把 embedder/reranker/llm 的真实 usage 桥接到 sink，
        # 关闭时传 None，采集侧 if 判空直接跳过（零开销）。
        monitoring = settings.monitoring
        token_usage_cb = self._record_token_usage if monitoring.enabled else None

        embedding_key = (
            settings.embedding.api_key.get_secret_value()
            if settings.embedding.api_key is not None
            else None
        )
        self.embedder = CredentialConfiguredEmbedder(
            async_session_factory,
            settings.embedding,
            expected_dimensions=settings.milvus.dense_dim,
            on_usage=token_usage_cb,
        )
        self.search_store = Milvus3SearchStore(settings.milvus)
        self.symbol_search_store = SymbolSearchStore(
            async_session_factory,
            max_scope_blobs=settings.retrieval.exact_max_scope_blobs,
            timeout_seconds=settings.retrieval.exact_timeout_seconds,
        )

        self.reranker = CredentialConfiguredReranker(
            async_session_factory,
            settings.rerank,
            fallback_embedding_key=embedding_key,
            on_usage=token_usage_cb,
        )

        # Initialize path index for filename queries
        self.path_index = None
        if settings.retrieval.path_index_enabled:
            try:
                self.path_index = PathIndexClient(settings.milvus)
                logger.info("Path index enabled for filename queries")
            except Exception as e:
                logger.warning("Failed to initialize path index: {}", e)
                self.path_index = None

        # LLM 三类（LLM 重排 / 查询改写 / 意图分类）各自按 kind 从 model_credentials 解析
        # 凭证（env 兜底），不再共用单一 client，可分别配置 key/base_url/model/tpm。
        # reload 命令会一并刷新它们（credential_runtime 持有列表）。
        llm_clients: list[CredentialConfiguredLLMClient] = []

        self.llm_reranker = None
        if settings.llm.rerank_enabled:
            rerank_llm = CredentialConfiguredLLMClient(
                "llm_rerank",
                async_session_factory,
                settings.llm,
                fallback_model=settings.llm.model,
                on_usage=token_usage_cb,
            )
            llm_clients.append(rerank_llm)
            from oce.domain.services.llm.reranker import LLMReranker
            self.llm_reranker = LLMReranker(
                client=rerank_llm,
                model=settings.llm.model,
                max_candidates=settings.llm.max_candidates,
                output_top_k=settings.llm.output_top_k,
                snippet_chars=settings.llm.snippet_chars,
            )
            logger.info("LLM reranker enabled (kind=llm_rerank)")

        self.query_rewriter = None
        if settings.retrieval.query_rewrite_enabled:
            rewrite_llm = CredentialConfiguredLLMClient(
                "query_rewrite",
                async_session_factory,
                settings.llm,
                fallback_model=settings.retrieval.query_rewrite_model,
                on_usage=token_usage_cb,
            )
            llm_clients.append(rewrite_llm)
            from oce.domain.services.llm.rewriter import QueryRewriter
            self.query_rewriter = QueryRewriter(
                client=rewrite_llm,
                model=settings.retrieval.query_rewrite_model,
                num_rewrites=settings.retrieval.query_rewrite_num,
            )
            logger.info("Query rewriter enabled (kind=query_rewrite)")

        self.intent_classifier = None
        if settings.retrieval.intent_classification_enabled:
            intent_llm = CredentialConfiguredLLMClient(
                "intent",
                async_session_factory,
                settings.llm,
                fallback_model=settings.llm.model,
                on_usage=token_usage_cb,
            )
            llm_clients.append(intent_llm)
            from oce.domain.services.llm.intent import IntentClassifier
            self.intent_classifier = IntentClassifier(
                llm_client=intent_llm,
                model=settings.llm.model,
            )
            logger.info("Intent classifier enabled (kind=intent)")

        credential_runtime = _CredentialRuntime(
            self.embedder, self.reranker, llm_clients
        )

        self.chunker = build_chunker()
        self._uow_factory = lambda: SqlAlchemyUnitOfWork(async_session_factory)

        # 监控 sink：启用时异步落库，否则空实现（monitoring 已在前面解析）
        if monitoring.enabled:
            self.metrics = SqlMetricsSink(
                async_session_factory,
                flush_interval_seconds=monitoring.flush_interval_seconds,
                max_buffer=monitoring.flush_max_buffer,
            )
        else:
            self.metrics = NoopMetricsSink()

        # 资源采样器：监控开启且 psutil 可用时后台周期采样，否则禁用
        if monitoring.enabled:
            self.resource_sampler = ResourceSampler(
                self.metrics,
                interval_seconds=monitoring.resource_sample_interval_seconds,
                collector=build_psutil_collector(os.environ.get(DATA_DIR_ENV)),
            )
        else:
            self.resource_sampler = None

        # 监控数据清理：监控开启时按 retention_days 周期清过期行（GC 另作独立流程）
        if monitoring.enabled:
            self.monitoring_cleaner = MonitoringCleaner(
                async_session_factory,
                retention_days=monitoring.retention_days,
                interval_seconds=monitoring.cleanup_interval_seconds,
            )
        else:
            self.monitoring_cleaner = None

        # 初始化 Redis 队列和 Worker（可选）
        self.queue = None
        self.worker = None
        if settings.worker.enabled:
            import redis.asyncio as redis
            redis_client = redis.from_url(
                settings.redis.url,
                decode_responses=True,
                encoding="utf-8",
                max_connections=20,  # 连接池大小（8 workers + 余量）
                socket_timeout=10.0,  # socket 超时 10 秒
                socket_connect_timeout=5.0,  # 连接超时 5 秒
                socket_keepalive=True,  # TCP keepalive
                health_check_interval=30,  # 健康检查间隔 30 秒
                retry_on_timeout=True,  # 超时自动重试
            )
            self.queue = RedisQueue(redis_client, settings.redis.queue_name)

            db_worker_capacity = max(1, settings.database.pool_size - 1)
            worker_concurrency = min(
                settings.worker.concurrency,
                db_worker_capacity,
                settings.embedding.max_concurrency,
            )
            if worker_concurrency != settings.worker.concurrency:
                logger.warning(
                    "Worker concurrency reduced from {} to {} to match DB and embedding limits",
                    settings.worker.concurrency,
                    worker_concurrency,
                )

            self.worker = EmbedWorker(
                queue=self.queue,
                uow_factory=self._uow_factory,
                chunker=self.chunker,
                embedder=self.embedder,
                vector_index=self.search_store,
                path_store=self.path_index,
                concurrency=worker_concurrency,
                max_retries=settings.worker.max_retries,
            )

        command_bus = CommandBus()
        command_bus.register(
            IngestBlobCommand,
            IngestBlobCommandHandler(
                self._uow_factory,
                self.chunker,
                self.embedder,
                self.search_store,
                self.queue,
            ),
        )
        command_bus.register(
            IngestBlobsCommand,
            IngestBlobsCommandHandler(
                self._uow_factory,
                self.chunker,
                self.embedder,
                self.search_store,
                self.queue,
            ),
        )
        command_bus.register(
            EmbedPendingCommand,
            EmbedPendingCommandHandler(
                self._uow_factory,
                self.chunker,
                self.embedder,
                self.search_store,
                path_store=self.path_index,
                blob_batch_size=32,
            ),
        )
        delete_blobs_handler = DeleteBlobsCommandHandler(
            self._uow_factory,
            self.search_store,
            path_store=self.path_index,
        )
        command_bus.register(DeleteBlobsCommand, delete_blobs_handler)
        command_bus.register(
            ReloadEmbeddingCredentialsCommand,
            ReloadEmbeddingCredentialsCommandHandler(credential_runtime),
        )
        command_bus.register(
            CheckpointCommand,
            CheckpointCommandHandler(self._uow_factory),
        )
        command_bus.register(
            RequeueStaleCommand,
            RequeueStaleCommandHandler(self._uow_factory, self.queue),
        )
        command_bus.register(
            ResetQueueCommand,
            ResetQueueCommandHandler(self._uow_factory, self.queue),
        )

        credential_admin_store = SqlCredentialAdminStore(async_session_factory)
        command_bus.register(
            CreateCredentialCommand,
            CreateCredentialCommandHandler(credential_admin_store),
        )
        command_bus.register(
            UpdateCredentialCommand,
            UpdateCredentialCommandHandler(credential_admin_store),
        )
        command_bus.register(
            DeleteCredentialCommand,
            DeleteCredentialCommandHandler(credential_admin_store),
        )
        command_bus.register(
            DuplicateCredentialCommand,
            DuplicateCredentialCommandHandler(credential_admin_store),
        )
        command_bus.register(
            GcCommand,
            GcCommandHandler(self._uow_factory, delete_blobs_handler, self.queue),
        )

        query_bus = QueryBus()
        query_bus.register(
            SearchQuery,
            SearchQueryHandler(
                RetrievalPipeline(
                    embedder=self.embedder,
                    store=self.search_store,
                    reranker=self.reranker,
                    llm_reranker=self.llm_reranker,
                    query_rewriter=self.query_rewriter,
                    path_store=self.path_index,
                    exact_store=self.symbol_search_store,
                    intent_classifier=self.intent_classifier,
                    settings=settings.retrieval,
                ),
                metrics=self.metrics,
                retrieval_audit_enabled=(
                    monitoring.enabled and monitoring.retrieval_audit_enabled
                ),
                store_query_text=monitoring.store_query_text,
            ),
        )
        query_bus.register(FindMissingQuery, FindMissingQueryHandler(self._uow_factory))
        query_bus.register(BlobStatusQuery, BlobStatusQueryHandler(self._uow_factory))
        query_bus.register(ResolveScopeQuery, ResolveScopeQueryHandler(self._uow_factory))
        query_bus.register(
            MonitoringStatsQuery,
            MonitoringStatsQueryHandler(
                SqlMonitoringStatsReader(async_session_factory)
            ),
        )
        query_bus.register(
            ListCredentialsQuery,
            ListCredentialsQueryHandler(credential_admin_store),
        )
        query_bus.register(
            QueueStatusQuery,
            QueueStatusQueryHandler(self._uow_factory, self.queue),
        )

        self.command_bus = command_bus
        self.query_bus = query_bus
        self.application = RetrievalApplication(
            command_bus,
            query_bus,
            background_indexing=self.queue is not None,
        )

    async def _record_token_usage(
        self,
        credential_id: int,
        kind: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """把 embedder/reranker/llm 的真实用量桥接到 sink。

        credential_id=0（无凭证，如 LLM）归一为 None；旁路容错：任何异常只记日志，
        绝不抛回主链路。
        """
        try:
            self.metrics.record_token_usage(
                TokenUsageRecord(
                    kind=kind,
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    credential_id=credential_id or None,
                )
            )
        except Exception as exc:  # 监控旁路：绝不影响主链路
            logger.warning("record token usage failed: {}", exc)

    async def close(self) -> None:
        if self.worker is not None:
            await self.worker.stop()
        if self.resource_sampler is not None:
            await self.resource_sampler.stop()
        if self.monitoring_cleaner is not None:
            await self.monitoring_cleaner.stop()
        await self.metrics.stop()
        await self.search_store.close()
        await self.embedder.close()
        await self.reranker.close()


@lru_cache
def get_container() -> Container:
    return Container()
