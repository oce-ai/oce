"""进程级 composition root。"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

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
from oce.application.commands.reconfigure import (
    ReconfigureRetrievalCommand,
    ReconfigureRetrievalCommandHandler,
    RetrievalConfigQuery,
    RetrievalConfigQueryHandler,
    RetrievalReconfigurator,
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
from oce.application.queries.reports import (
    ApiCallsReportQuery,
    ApiCallsReportQueryHandler,
    EmptyQueriesQuery,
    EmptyQueriesQueryHandler,
    IndexInventoryQuery,
    IndexInventoryQueryHandler,
    ResourcesReportQuery,
    ResourcesReportQueryHandler,
    RetrievalReportQuery,
    RetrievalReportQueryHandler,
    SlowQueriesQuery,
    SlowQueriesQueryHandler,
    StorageReportQuery,
    StorageReportQueryHandler,
    TokensReportQuery,
    TokensReportQueryHandler,
)
from oce.application.queries.search import SearchQuery
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
from oce.application.factories.retrieval import RetrievalDeps, build_retrieval_stack
from oce.application.worker import EmbedWorker
from oce.infrastructure.embed.credential_embedder import CredentialConfiguredEmbedder
from oce.infrastructure.embed.credential_reranker import CredentialConfiguredReranker
from oce.infrastructure.milvus3 import Milvus3SearchStore
from oce.infrastructure.milvus3.path_index import PathIndexClient
from oce.infrastructure.persistence.credential_admin_store import (
    SqlCredentialAdminStore,
)
from oce.infrastructure.persistence.uow import SqlAlchemyUnitOfWork
from oce.infrastructure.metrics.cleanup import MonitoringCleaner
from oce.infrastructure.metrics.resource_sampler import (
    ResourceSampler,
    build_psutil_collector,
)
from oce.infrastructure.metrics.sql_metrics_sink import SqlMetricsSink
from oce.infrastructure.metrics.stats_store import SqlMonitoringStatsReader
from oce.infrastructure.metrics.report_store import SqlReportsReader
from oce.infrastructure.queue.redis_queue import RedisQueue
from oce.shared.config import get_settings
from oce.shared.database.session import async_session_factory
from oce.shared.logging import DATA_DIR_ENV
from oce.shared.metrics import NoopMetricsSink, TokenUsageRecord
from oce.shared.reports_read import VectorCollectionStat, VectorStoreStat


class _CredentialRuntime:
    def __init__(self, embedder, reranker, llm_clients=()) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self._llm_clients = [client for client in llm_clients if client is not None]

    def set_llm_clients(self, llm_clients) -> None:
        """热重载换栈后同步 LLM client 覆盖面（llm_rerank 开关会增减 client）。

        必须在 query_bus 原子重注册之后调用：reload 命令据此刷新当前 in-force 的
        client 集合，否则会对已退休的旧 client 调 reload、漏掉新建的。
        """
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

        # 监控 sink 先于检索栈构造：工厂需要它作为 SearchQueryHandler 的审计落点。
        # 纯构造无副作用（缓冲是 deque，落库 task 在 lifespan 调 start() 才建），
        # 前移不改变启动语义。
        if monitoring.enabled:
            self.metrics = SqlMetricsSink(
                async_session_factory,
                flush_interval_seconds=monitoring.flush_interval_seconds,
                max_buffer=monitoring.flush_max_buffer,
            )
        else:
            self.metrics = NoopMetricsSink()

        # LLM 三类（LLM 重排 / 查询改写 / 意图分类）各自按 kind 从 model_credentials 解析
        # 凭证（env 兜底），不再共用单一 client，可分别配置 key/base_url/model/tpm。
        # reload 命令会一并刷新它们（credential_runtime 持有列表）。
        # 检索栈（pipeline + handler + symbol store + LLM 三件套）统一由工厂构造，
        # 使冷启动与热重载（RetrievalReconfigurator）共用同一条构造路径。
        retrieval_deps = RetrievalDeps(
            embedder=self.embedder,
            search_store=self.search_store,
            base_reranker=self.reranker,
            path_index=self.path_index,
            session_factory=async_session_factory,
            metrics=self.metrics,
            retrieval_audit_enabled=(
                monitoring.enabled and monitoring.retrieval_audit_enabled
            ),
            store_query_text=monitoring.store_query_text,
            token_usage_cb=token_usage_cb,
            llm_settings=settings.llm,
        )
        self._retrieval_deps = retrieval_deps
        stack = build_retrieval_stack(settings, retrieval_deps)
        self.symbol_search_store = stack.symbol_store
        # 以下三项仅为兼容既有读取点保留；pipeline 实际持有的是 stack 内部实例。
        self.llm_reranker = stack.pipeline.llm_reranker
        self.query_rewriter = stack.pipeline.query_rewriter
        self.intent_classifier = stack.pipeline.intent_classifier
        self._retrieval_stack = stack

        credential_runtime = _CredentialRuntime(
            self.embedder, self.reranker, list(stack.llm_clients)
        )

        self.chunker = build_chunker()
        self._uow_factory = lambda: SqlAlchemyUnitOfWork(async_session_factory)

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
        # handler 由工厂构造（与热重载共用同一路径），此处只负责注册。
        query_bus.register(SearchQuery, stack.handler)
        query_bus.register(FindMissingQuery, FindMissingQueryHandler(self._uow_factory))
        query_bus.register(BlobStatusQuery, BlobStatusQueryHandler(self._uow_factory))
        query_bus.register(ResolveScopeQuery, ResolveScopeQueryHandler(self._uow_factory))
        query_bus.register(
            MonitoringStatsQuery,
            MonitoringStatsQueryHandler(
                SqlMonitoringStatsReader(async_session_factory)
            ),
        )
        # 报表 reader 共享一个实例：只读聚合，无状态，8 个 handler 复用。
        # 向量库统计以异步闭包注入，让 reader 保持纯 SQL 模块不依赖 milvus infra；
        # 异常在 reader 侧统一降级为 mode="unavailable"，这里放心 propagate。
        milvus_settings = settings.milvus
        search_store = self.search_store

        async def _vector_stats() -> VectorStoreStat:
            local = not milvus_settings.endpoint.startswith(("http://", "https://"))
            rows = await search_store.client.collection_stats()
            collections = tuple(
                VectorCollectionStat(
                    name=name,
                    rows=count,
                    est_bytes=count * milvus_settings.dense_dim * 4,
                )
                for name, count in rows
            )
            file_bytes = 0
            if local:
                lite_path = Path(milvus_settings.endpoint)
                if lite_path.is_file():
                    file_bytes = lite_path.stat().st_size
            return VectorStoreStat(
                mode="lite" if local else "server",
                collections=collections,
                file_bytes=file_bytes,
            )

        reports_reader = SqlReportsReader(
            async_session_factory,
            data_dir=os.environ.get(DATA_DIR_ENV),
            vector_stats=_vector_stats,
        )
        query_bus.register(
            ApiCallsReportQuery, ApiCallsReportQueryHandler(reports_reader)
        )
        query_bus.register(
            RetrievalReportQuery, RetrievalReportQueryHandler(reports_reader)
        )
        query_bus.register(SlowQueriesQuery, SlowQueriesQueryHandler(reports_reader))
        query_bus.register(EmptyQueriesQuery, EmptyQueriesQueryHandler(reports_reader))
        query_bus.register(TokensReportQuery, TokensReportQueryHandler(reports_reader))
        query_bus.register(
            IndexInventoryQuery, IndexInventoryQueryHandler(reports_reader)
        )
        query_bus.register(
            ResourcesReportQuery, ResourcesReportQueryHandler(reports_reader)
        )
        query_bus.register(StorageReportQuery, StorageReportQueryHandler(reports_reader))
        query_bus.register(
            ListCredentialsQuery,
            ListCredentialsQueryHandler(credential_admin_store),
        )
        query_bus.register(
            QueueStatusQuery,
            QueueStatusQueryHandler(self._uow_factory, self.queue),
        )

        # L0 检索参数热重载器：复用同一份 deps（昂贵对象不重建）+ query_bus（原子重注册
        # 落点）+ live settings（milvus/rerank 就地写回的目标）。generation 从 0 起。
        def _on_swap(new_stack) -> None:
            # 热重载换栈后，让既有读取点继续指向 in-force 实例（与冷启动同源）。
            self.symbol_search_store = new_stack.symbol_store
            self.llm_reranker = new_stack.pipeline.llm_reranker
            self.query_rewriter = new_stack.pipeline.query_rewriter
            self.intent_classifier = new_stack.pipeline.intent_classifier
            self._retrieval_stack = new_stack

        self.reconfigurator = RetrievalReconfigurator(
            settings=settings,
            deps=retrieval_deps,
            query_bus=query_bus,
            credential_runtime=credential_runtime,
            on_swap=_on_swap,
        )
        command_bus.register(
            ReconfigureRetrievalCommand,
            ReconfigureRetrievalCommandHandler(self.reconfigurator),
        )
        query_bus.register(
            RetrievalConfigQuery,
            RetrievalConfigQueryHandler(self.reconfigurator),
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
