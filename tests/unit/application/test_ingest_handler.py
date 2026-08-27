"""索引命令处理器测试。"""

from __future__ import annotations

import pytest

from oce.application.commands.ingest import (
    DeleteBlobsCommand,
    DeleteBlobsCommandHandler,
    EmbedPendingCommand,
    EmbedPendingCommandHandler,
    IngestBlobCommand,
    IngestBlobCommandHandler,
)
from oce.domain.blob.blob import Blob, BlobStatus
from oce.domain.chunk import Chunk, RecursiveChunker, LocatedChunk
from oce.domain.services.indexing import IndexingPipeline

from tests.unit.application.fakes import (
    FakeEmbedder,
    FakeSearchStore,
    FakeUnitOfWorkFactory,
    blob_name,
)


@pytest.fixture
def dependencies():
    factory = FakeUnitOfWorkFactory()
    chunker = RecursiveChunker(chunk_size=6000, chunk_overlap=200)
    embedder = FakeEmbedder()
    index = FakeSearchStore()
    return factory, chunker, embedder, index


class RecordingQueue:
    def __init__(self, factory):
        self.factory = factory
        self.enqueued: list[str] = []
        self.commit_counts: list[int] = []

    async def enqueue(self, blob_name: str) -> None:
        self.enqueued.append(blob_name)
        self.commit_counts.append(self.factory.uow.commits)


async def test_ingest_returns_blob_name_and_count(dependencies):
    """异步模式：ingest 返回 0，embed_pending 完成切块后才有 chunk 数量"""
    factory, chunker, embedder, index = dependencies
    content = "\n".join(f"line{i}" for i in range(100))
    name = blob_name("src/a.py", content)
    handler = IngestBlobCommandHandler(factory, chunker, embedder, index)

    result = await handler.handle(IngestBlobCommand(name, "src/a.py", content))

    assert result.blob_name == name
    assert result.chunk_count == 0  # 异步模式返回 0
    assert factory.uow.blobs.blobs[name].status == BlobStatus.PENDING
    assert factory.uow.blobs.staging[name] == content
    assert factory.uow.commits == 1


async def test_ingest_enqueues_pending_blob_after_commit(dependencies):
    factory, chunker, embedder, index = dependencies
    queue = RecordingQueue(factory)
    content = "print('queued')"
    name = blob_name("src/queued.py", content)
    handler = IngestBlobCommandHandler(factory, chunker, embedder, index, queue)

    await handler.handle(IngestBlobCommand(name, "src/queued.py", content))

    assert queue.enqueued == [name]
    assert queue.commit_counts == [1]


async def test_ingest_does_not_stage_or_enqueue_ignored_blob(dependencies):
    factory, chunker, embedder, index = dependencies
    queue = RecordingQueue(factory)
    content = '{"version": 3}'
    name = blob_name("dist/app.js.map", content)
    handler = IngestBlobCommandHandler(factory, chunker, embedder, index, queue)

    await handler.handle(IngestBlobCommand(name, "dist/app.js.map", content))

    assert factory.uow.blobs.blobs[name].status == BlobStatus.READY
    assert name not in factory.uow.blobs.staging
    assert queue.enqueued == []


async def test_ingest_blank_content_is_ready(dependencies):
    """空内容在 embed_pending 后直接标记 READY"""
    factory, chunker, embedder, index = dependencies
    content = "\n\n   \n"
    name = blob_name("src/blank.py", content)

    # ingest 返回 PENDING
    ingest = IngestBlobCommandHandler(factory, chunker, embedder, index)
    result = await ingest.handle(IngestBlobCommand(name, "src/blank.py", content))
    assert result.chunk_count == 0
    assert factory.uow.blobs.blobs[name].status == BlobStatus.PENDING

    # embed_pending 处理空内容，标记 READY
    handler = EmbedPendingCommandHandler(factory, chunker, embedder, index)
    await handler.handle(EmbedPendingCommand((name,)))
    assert factory.uow.blobs.blobs[name].status == BlobStatus.READY


async def test_embed_pending_writes_vector_and_marks_ready(dependencies):
    """embed_pending 完成切块、嵌入，并标记 blob 为 ready"""
    factory, chunker, embedder, index = dependencies
    content = "print('hello')"
    path = "src/hello.py"
    name = blob_name(path, content)

    # 先 ingest 写元数据和 staging
    ingest = IngestBlobCommandHandler(factory, chunker, embedder, index)
    await ingest.handle(IngestBlobCommand(name, path, content))

    # embed_pending 完成切块和嵌入
    handler = EmbedPendingCommandHandler(factory, chunker, embedder, index)
    result = await handler.handle(EmbedPendingCommand((name,)))

    assert result.embedded_count == 1
    assert factory.uow.blobs.blobs[name].status == BlobStatus.READY
    assert len(index.upserted) == 1
    assert index.upserted[0]["blob_name"] == name


async def test_embed_pending_limits_vector_batches(dependencies):
    """embed_pending 成功处理多块内容"""
    factory, chunker, embedder, index = dependencies
    path = "src/large.py"
    # 生成足够大的内容确保切成多块（每行约 10 字符，1000 行 > 6000 chunk_size）
    content = "\n".join(f"line{i}" for i in range(1000))
    name = blob_name(path, content)

    # ingest + embed_pending
    ingest = IngestBlobCommandHandler(factory, chunker, embedder, index)
    await ingest.handle(IngestBlobCommand(name, path, content))

    handler = EmbedPendingCommandHandler(factory, chunker, embedder, index)
    result = await handler.handle(EmbedPendingCommand((name,)))

    assert result.embedded_count > 1  # 验证确实切了多块
    assert factory.uow.blobs.blobs[name].status == BlobStatus.READY


async def test_embed_failure_commits_error_state(dependencies):
    """嵌入失败时，blob 状态标记为 ERROR"""
    class FailingEmbedder:
        async def embed_documents(self, _texts):
            raise RuntimeError("provider failed")

    factory, chunker, _, index = dependencies
    # 用足够长的内容保证会切块
    content = "\n".join(f"print({i})" for i in range(50))
    path = "src/broken.py"
    name = blob_name(path, content)

    # ingest 正常完成
    await IngestBlobCommandHandler(
        factory, chunker, FakeEmbedder(), index
    ).handle(IngestBlobCommand(name, path, content))

    # embed_pending 使用失败的 embedder
    handler = EmbedPendingCommandHandler(
        factory, chunker, FailingEmbedder(), index
    )

    with pytest.raises(RuntimeError, match="provider failed"):
        await handler.handle(EmbedPendingCommand((name,)))

    assert factory.uow.blobs.blobs[name].status == BlobStatus.ERROR
    assert factory.uow.commits == 2  # ingest + error commit


async def test_delete_commits_metadata_before_deleting_vectors(dependencies):
    factory, _, _, index = dependencies
    name = blob_name("src/deleted.py", "content")
    factory.uow.blobs.blobs[name] = Blob(
        blob_name=name,
        path="src/deleted.py",
        status=BlobStatus.READY,
    )

    await DeleteBlobsCommandHandler(factory, index).handle(
        DeleteBlobsCommand((name,))
    )

    assert name not in factory.uow.blobs.blobs
    assert factory.uow.commits == 1
    assert index.deleted == [name]
