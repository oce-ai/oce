"""Worker retry-state tests."""

from __future__ import annotations

from oce.application.worker import EmbedWorker
from oce.domain.blob.blob import BlobStatus
from oce.domain.chunk import RecursiveChunker
from tests.unit.application.fakes import (
    FakeSearchStore,
    FakeUnitOfWorkFactory,
    blob_name,
)


class FailingEmbedder:
    async def embed_documents(self, _texts):
        raise RuntimeError("provider failed")


class RetryQueue:
    def __init__(self, blob_name: str) -> None:
        self.blob_name = blob_name
        self.worker = None
        self.dequeue_count = 0
        self.failed: list[str] = []
        self.enqueued: list[str] = []

    async def dequeue(self, timeout=5):
        self.dequeue_count += 1
        if self.dequeue_count == 1:
            return self.blob_name
        self.worker._running = False
        return None

    async def ack(self, _blob_name: str) -> None:
        raise AssertionError("failed embedding must not be acknowledged")

    async def fail(self, blob_name: str) -> None:
        self.failed.append(blob_name)

    async def enqueue(self, blob_name: str) -> None:
        self.enqueued.append(blob_name)


async def _run_failure(max_retries: int):
    factory = FakeUnitOfWorkFactory()
    path = "src/failing.py"
    content = "print('failing')"
    name = blob_name(path, content)
    from oce.application.commands.ingest import IngestBlobCommand, IngestBlobCommandHandler

    await IngestBlobCommandHandler(
        factory,
        RecursiveChunker(),
        FailingEmbedder(),
        FakeSearchStore(),
    ).handle(IngestBlobCommand(name, path, content))
    queue = RetryQueue(name)
    worker = EmbedWorker(
        queue=queue,
        uow_factory=factory,
        chunker=RecursiveChunker(),
        embedder=FailingEmbedder(),
        vector_index=FakeSearchStore(),
        max_retries=max_retries,
    )
    queue.worker = worker
    worker._running = True
    await worker._loop(0)
    return factory, queue, name


async def test_worker_requeues_pending_blob_before_retry_limit():
    factory, queue, name = await _run_failure(max_retries=2)

    blob = factory.uow.blobs.blobs[name]
    assert blob.status == BlobStatus.PENDING
    assert blob.retry_count == 1
    assert queue.failed == [name]
    assert queue.enqueued == [name]
    assert name in factory.uow.blobs.staging


async def test_worker_marks_error_and_cleans_staging_at_retry_limit():
    factory, queue, name = await _run_failure(max_retries=1)

    blob = factory.uow.blobs.blobs[name]
    assert blob.status == BlobStatus.ERROR
    assert blob.error_message == "provider failed"
    assert queue.enqueued == []
    assert name not in factory.uow.blobs.staging