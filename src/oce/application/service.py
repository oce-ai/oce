"""对 HTTP、CLI 和评测器提供稳定的 application API。"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from oce.application.bus import CommandBus, QueryBus
from oce.application.commands.checkpoint import CheckpointCommand, CheckpointResult
from oce.application.commands.credentials import (
    ReloadEmbeddingCredentialsCommand,
    ReloadEmbeddingCredentialsResult,
)
from oce.application.commands.ingest import (
    EmbedPendingCommand,
    IngestBlobCommand,
    IngestBlobsCommand,
)
from oce.application.commands.gc import GcCommand, GcResult
from oce.application.commands.queue_admin import ResetQueueCommand, ResetQueueResult
from oce.application.commands.requeue import RequeueStaleCommand, RequeueStaleResult
from oce.application.credential_admin import (
    CreateCredentialCommand,
    CredentialCreate,
    CredentialDuplicate,
    CredentialRecord,
    CredentialUpdate,
    DeleteCredentialCommand,
    DuplicateCredentialCommand,
    ListCredentialsQuery,
    UpdateCredentialCommand,
)
from oce.application.queries.queue import QueueStatusQuery, QueueStatusResult
from oce.application.queries.reports import (
    ApiCallsReportQuery,
    EmptyQueriesQuery,
    IndexInventoryQuery,
    ResourcesReportQuery,
    RetrievalReportQuery,
    SlowQueriesQuery,
    StorageReportQuery,
    TokensReportQuery,
)
from oce.application.queries.search import SearchQuery
from oce.application.queries.stats import MonitoringStatsQuery
from oce.application.queries.status import (
    BlobStatusQuery,
    BlobStatusResult,
    FindMissingQuery,
    FindMissingResult,
    ResolveScopeQuery,
    ResolveScopeResult,
)
from oce.domain.services.formatter import format_retrieval
from oce.domain.services.search import SearchHit
from oce.shared.metrics_read import MonitoringStats
from oce.shared.reports_read import (
    ApiCallsReport,
    IndexInventoryReport,
    ResourcesReport,
    RetrievalQueryDetail,
    RetrievalReport,
    StorageReport,
    TokensReport,
)


def compute_blob_name(path: str, content: str) -> str:
    """生成与 ACE 客户端一致的内容地址：``sha256(path + content)``。"""
    return hashlib.sha256(f"{path}{content}".encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BlobUpload:
    path: str
    content: str


@dataclass(frozen=True)
class BatchUploadResult:
    blob_names: tuple[str, ...]
    chunk_count: int
    embedded_count: int


@dataclass(frozen=True)
class RetrievalResult:
    hits: tuple[SearchHit, ...]
    formatted_retrieval: str
    elapsed_ms: int


class RetrievalApplication:
    """跨命令/查询的用例编排；传输层只负责 DTO 映射。"""

    def __init__(
        self,
        command_bus: CommandBus,
        query_bus: QueryBus,
        *,
        background_indexing: bool = False,
    ) -> None:
        self._commands = command_bus
        self._queries = query_bus
        self._background_indexing = background_indexing

    async def find_missing(self, blob_names: list[str]) -> FindMissingResult:
        return await self._queries.ask(FindMissingQuery(tuple(blob_names)))

    async def reload_embedding_credentials(
        self,
    ) -> ReloadEmbeddingCredentialsResult:
        return await self._commands.execute(ReloadEmbeddingCredentialsCommand())

    async def batch_upload(
        self,
        blobs: list[BlobUpload],
        *,
        checkpoint_id: str | None = None,
    ) -> BatchUploadResult:
        commands = tuple(
            IngestBlobCommand(
                compute_blob_name(blob.path, blob.content),
                blob.path,
                blob.content,
            )
            for blob in blobs
        )
        names = [command.blob_name for command in commands]
        result = await self._commands.execute(IngestBlobsCommand(commands))
        embedded_count = 0
        if not self._background_indexing:
            embedded = await self._commands.execute(EmbedPendingCommand(tuple(names)))
            embedded_count = embedded.embedded_count
        if checkpoint_id:
            # 可选：上传内容索引后，把本次 blob 直接登记进已有 checkpoint 链
            # （CheckpointCommand 对非空 checkpoint_id 只推进已有链，不会隐式创建）
            await self._commands.execute(
                CheckpointCommand(checkpoint_id, tuple(names), ())
            )
        return BatchUploadResult(tuple(names), result.chunk_count, embedded_count)

    async def retrieve(
        self,
        information_request: str,
        *,
        checkpoint_id: str | None = None,
        added_blobs: list[str] | None = None,
        deleted_blobs: list[str] | None = None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        added = tuple(added_blobs or ())
        deleted = tuple(deleted_blobs or ())
        scope = await self._prepare_scope(checkpoint_id, added, deleted)
        result = await self._queries.ask(
            SearchQuery(information_request, scope.blob_names, source="retrieval")
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RetrievalResult(
            hits=tuple(result.hits),
            formatted_retrieval=format_retrieval(result.hits),
            elapsed_ms=elapsed_ms,
        )

    async def _prepare_scope(
        self,
        checkpoint_id: str | None,
        added: tuple[str, ...],
        deleted: tuple[str, ...],
    ) -> ResolveScopeResult:
        # 查询路径对 deleted_blobs 无副作用：只做本次检索范围差集，不删除任何服务端
        # 数据。物理清理由独立 GC 流程负责。
        if added and not self._background_indexing:
            await self._commands.execute(EmbedPendingCommand(added))
        return await self._queries.ask(ResolveScopeQuery(checkpoint_id, added, deleted))

    async def checkpoint(
        self,
        *,
        checkpoint_id: str | None,
        added_blobs: list[str],
        deleted_blobs: list[str],
    ) -> CheckpointResult:
        return await self._commands.execute(
            CheckpointCommand(
                checkpoint_id,
                tuple(added_blobs),
                tuple(deleted_blobs),
            )
        )

    async def blob_status(
        self,
        *,
        blob_names: list[str],
        checkpoint_id: str | None,
    ) -> BlobStatusResult:
        return await self._queries.ask(
            BlobStatusQuery(tuple(blob_names), checkpoint_id)
        )

    async def monitoring_stats(self, *, window_hours: int = 24) -> MonitoringStats:
        return await self._queries.ask(MonitoringStatsQuery(window_hours))

    async def api_calls_report(
        self, *, window_hours: int = 24, bucket: str = "hour"
    ) -> ApiCallsReport:
        return await self._queries.ask(ApiCallsReportQuery(window_hours, bucket))

    async def retrieval_report(
        self, *, window_hours: int = 24, bucket: str = "hour"
    ) -> RetrievalReport:
        return await self._queries.ask(RetrievalReportQuery(window_hours, bucket))

    async def slow_queries_report(
        self, *, window_hours: int = 24, limit: int = 50
    ) -> tuple[RetrievalQueryDetail, ...]:
        return await self._queries.ask(SlowQueriesQuery(window_hours, limit))

    async def empty_queries_report(
        self, *, window_hours: int = 24, limit: int = 50
    ) -> tuple[RetrievalQueryDetail, ...]:
        return await self._queries.ask(EmptyQueriesQuery(window_hours, limit))

    async def tokens_report(
        self, *, window_hours: int = 24, bucket: str = "hour"
    ) -> TokensReport:
        return await self._queries.ask(TokensReportQuery(window_hours, bucket))

    async def index_inventory_report(self) -> IndexInventoryReport:
        return await self._queries.ask(IndexInventoryQuery())

    async def resources_report(
        self, *, window_hours: int = 24, bucket: str = "hour"
    ) -> ResourcesReport:
        return await self._queries.ask(ResourcesReportQuery(window_hours, bucket))

    async def storage_report(self) -> StorageReport:
        return await self._queries.ask(StorageReportQuery())

    async def queue_status(self) -> QueueStatusResult:
        return await self._queries.ask(QueueStatusQuery())

    async def reset_queue(
        self, *, mode: str = "sync", requeue: bool = True
    ) -> ResetQueueResult:
        return await self._commands.execute(ResetQueueCommand(mode, requeue))

    async def requeue_stale(
        self, *, stale_hours: int = 24, limit: int = 100
    ) -> RequeueStaleResult:
        return await self._commands.execute(
            RequeueStaleCommand(stale_hours, limit)
        )

    async def run_gc(
        self, *, ttl_days: int = 30, dry_run: bool = True, limit: int = 1000
    ) -> GcResult:
        return await self._commands.execute(GcCommand(ttl_days, dry_run, limit))

    async def list_credentials(self) -> list[CredentialRecord]:
        return await self._queries.ask(ListCredentialsQuery())

    async def create_credential(self, data: CredentialCreate) -> CredentialRecord:
        return await self._commands.execute(CreateCredentialCommand(data))

    async def update_credential(
        self, credential_id: int, changes: CredentialUpdate
    ) -> CredentialRecord | None:
        return await self._commands.execute(
            UpdateCredentialCommand(credential_id, changes)
        )

    async def delete_credential(self, credential_id: int) -> bool:
        return await self._commands.execute(DeleteCredentialCommand(credential_id))

    async def duplicate_credential(
        self, credential_id: int, changes: CredentialDuplicate
    ) -> CredentialRecord | None:
        return await self._commands.execute(
            DuplicateCredentialCommand(credential_id, changes)
        )
