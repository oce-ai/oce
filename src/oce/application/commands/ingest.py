"""索引写路径命令。"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger

from oce.application.messages import Command
from oce.application.uow import UnitOfWorkFactory
from oce.domain.blob.blob import BlobStatus
from oce.domain.chunk import Chunker
from oce.domain.services.embedder import Embedder
from oce.domain.services.indexing import IndexingPipeline
from oce.domain.services.path_search import PathSearchStore
from oce.domain.services.search import VectorIndex


@dataclass(frozen=True)
class IngestBlobCommand(Command):
    blob_name: str
    path: str
    content: str


@dataclass(frozen=True)
class IngestBlobResult:
    blob_name: str
    chunk_count: int


class IngestBlobCommandHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        chunker: Chunker,
        embedder: Embedder,
        vector_index: VectorIndex,
        queue=None,  # 可选：启用异步时传入 RedisQueue
    ) -> None:
        self._uow_factory = uow_factory
        self._chunker = chunker
        self._embedder = embedder
        self._vector_index = vector_index
        self._queue = queue

    async def handle(self, command: IngestBlobCommand) -> IngestBlobResult:
        should_enqueue = False
        async with self._uow_factory() as uow:
            pipeline = IndexingPipeline(
                chunker=self._chunker,
                embedder=self._embedder,
                vector_index=self._vector_index,
                blob_repo=uow.blobs,
                chunk_repo=uow.chunks,
            )
            count = await pipeline.ingest(command.blob_name, command.path, command.content)
            stored = await uow.blobs.get(command.blob_name)
            should_enqueue = stored is not None and stored.status == BlobStatus.PENDING
            await uow.commit()
        if self._queue is not None and should_enqueue:
            await self._queue.enqueue(command.blob_name)
        return IngestBlobResult(command.blob_name, count)


@dataclass(frozen=True)
class IngestBlobsCommand(Command):
    blobs: tuple[IngestBlobCommand, ...]


@dataclass(frozen=True)
class IngestBlobsResult:
    chunk_count: int


class IngestBlobsCommandHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        chunker: Chunker,
        embedder: Embedder,
        vector_index: VectorIndex,
        queue=None,  # 可选：启用异步时传入 RedisQueue
    ) -> None:
        self._uow_factory = uow_factory
        self._chunker = chunker
        self._embedder = embedder
        self._vector_index = vector_index
        self._queue = queue

    async def handle(self, command: IngestBlobsCommand) -> IngestBlobsResult:
        if not command.blobs:
            return IngestBlobsResult(0)
        pending_names: list[str] = []
        async with self._uow_factory() as uow:
            pipeline = IndexingPipeline(
                chunker=self._chunker,
                embedder=self._embedder,
                vector_index=self._vector_index,
                blob_repo=uow.blobs,
                chunk_repo=uow.chunks,
            )
            chunk_count = 0
            for blob in command.blobs:
                chunk_count += await pipeline.ingest(
                    blob.blob_name,
                    blob.path,
                    blob.content,
                )
                stored = await uow.blobs.get(blob.blob_name)
                if stored is not None and stored.status == BlobStatus.PENDING:
                    pending_names.append(blob.blob_name)
            await uow.commit()
        if self._queue is not None:
            for blob_name in pending_names:
                await self._queue.enqueue(blob_name)
        return IngestBlobsResult(chunk_count)


@dataclass(frozen=True)
class EmbedPendingCommand(Command):
    blob_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class EmbedPendingResult:
    embedded_count: int


class EmbedPendingCommandHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        chunker: Chunker,
        embedder: Embedder,
        vector_index: VectorIndex,
        path_store: PathSearchStore | None = None,
        blob_batch_size: int = 32,
    ) -> None:
        if blob_batch_size < 1:
            raise ValueError("blob_batch_size must be positive")
        self._uow_factory = uow_factory
        self._chunker = chunker
        self._embedder = embedder
        self._vector_index = vector_index
        self._path_store = path_store
        self._blob_batch_size = blob_batch_size

    async def handle(self, command: EmbedPendingCommand) -> EmbedPendingResult:
        names = command.blob_names
        if names is None:
            groups: list[tuple[str, ...] | None] = [None]
        else:
            groups = [
                names[offset:offset + self._blob_batch_size]
                for offset in range(0, len(names), self._blob_batch_size)
            ]

        embedded = 0
        for group in groups:
            if group == ():
                continue
            async with self._uow_factory() as uow:
                pipeline = IndexingPipeline(
                    chunker=self._chunker,
                    embedder=self._embedder,
                    vector_index=self._vector_index,
                    blob_repo=uow.blobs,
                    chunk_repo=uow.chunks,
                    path_store=self._path_store,
                )
                try:
                    embedded += await pipeline.embed_pending(group)
                except Exception:
                    await uow.commit()
                    raise
                else:
                    await uow.commit()
        return EmbedPendingResult(embedded)


@dataclass(frozen=True)
class DeleteBlobsCommand(Command):
    blob_names: tuple[str, ...]


class DeleteBlobsCommandHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        vector_index: VectorIndex,
        path_store: PathSearchStore | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._vector_index = vector_index
        self._path_store = path_store

    async def handle(self, command: DeleteBlobsCommand) -> None:
        if not command.blob_names:
            return
        async with self._uow_factory() as uow:
            await uow.blobs.delete_many(command.blob_names)
            await uow.commit()
        await self._vector_index.delete(list(command.blob_names))
        if self._path_store is not None:
            try:
                await self._path_store.delete_by_blob_names(list(command.blob_names))
            except Exception as exc:
                # 路径索引删除失败不阻塞删除主流程，仅记日志
                logger.warning("path index delete failed for {} blobs: {}", len(command.blob_names), exc)
