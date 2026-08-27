"""对 HTTP、CLI 和评测器提供稳定的 application API。"""

from __future__ import annotations

import asyncio
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
    DeleteBlobsCommand,
    EmbedPendingCommand,
    IngestBlobCommand,
    IngestBlobsCommand,
)
from oce.application.queries.search import SearchQuery
from oce.application.queries.status import (
    BlobStatusQuery,
    BlobStatusResult,
    FindMissingQuery,
    FindMissingResult,
    KeyDocResult,
    OverviewContextQuery,
    ResolveScopeQuery,
    ResolveScopeResult,
)
from oce.domain.services.formatter import format_retrieval
from oce.domain.services.search import SearchHit
from oce.shared.errors import ServiceNotReadyError


_OVERVIEW_QUERIES = (
    "Where are the main application entry points and startup lifecycle implemented?",
    "What are the core abstractions and module boundaries?",
    "Where are the public APIs, protocols, and schemas defined?",
    "How are errors, logging, and observability handled?",
)
_DEEP_OVERVIEW_QUERIES = (
    "How is the project built, configured, and deployed?",
    "How are concurrency, queues, and background work coordinated?",
    "Where are the most important integration and end-to-end tests?",
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


@dataclass(frozen=True)
class ProjectOverviewSection:
    query: str
    formatted_retrieval: str = ""
    error: str | None = None


@dataclass(frozen=True)
class ProjectOverviewResult:
    key_docs: tuple[KeyDocResult, ...]
    sections: tuple[ProjectOverviewSection, ...]
    working_set_paths: tuple[str, ...]
    working_set_paths_total: int
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

    async def batch_upload(self, blobs: list[BlobUpload]) -> BatchUploadResult:
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
            SearchQuery(information_request, scope.blob_names)
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return RetrievalResult(
            hits=tuple(result.hits),
            formatted_retrieval=format_retrieval(result.hits),
            elapsed_ms=elapsed_ms,
        )

    async def project_overview(
        self,
        *,
        depth: str,
        checkpoint_id: str | None = None,
        added_blobs: list[str] | None = None,
        deleted_blobs: list[str] | None = None,
    ) -> ProjectOverviewResult:
        started = time.perf_counter()
        added = tuple(added_blobs or ())
        deleted = tuple(deleted_blobs or ())
        scope = await self._prepare_scope(checkpoint_id, added, deleted)
        queries = _OVERVIEW_QUERIES + (
            _DEEP_OVERVIEW_QUERIES if depth == "deep" else ()
        )

        async def run(query: str) -> ProjectOverviewSection:
            try:
                result = await self._queries.ask(SearchQuery(query, scope.blob_names))
                return ProjectOverviewSection(
                    query=query,
                    formatted_retrieval=format_retrieval(result.hits),
                )
            except ServiceNotReadyError:
                raise
            except Exception as exc:
                return ProjectOverviewSection(query=query, error=str(exc))

        context, sections = await asyncio.gather(
            self._queries.ask(
                OverviewContextQuery(scope.blob_names)
            ),
            asyncio.gather(*(run(query) for query in queries)),
        )
        return ProjectOverviewResult(
            key_docs=context.key_docs,
            sections=tuple(sections),
            working_set_paths=context.paths,
            working_set_paths_total=context.paths_total,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    async def _prepare_scope(
        self,
        checkpoint_id: str | None,
        added: tuple[str, ...],
        deleted: tuple[str, ...],
    ) -> ResolveScopeResult:
        if deleted:
            await self._commands.execute(DeleteBlobsCommand(deleted))
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
