"""Tests for transaction-batched ingestion."""

from __future__ import annotations

from oce.application.commands.ingest import (
    IngestBlobCommand,
    IngestBlobsCommand,
    IngestBlobsCommandHandler,
)
from oce.domain.chunk import RecursiveChunker
from tests.unit.application.fakes import (
    FakeEmbedder,
    FakeSearchStore,
    FakeUnitOfWorkFactory,
    blob_name,
)


async def test_ingest_blobs_commits_one_transaction_for_the_batch():
    """批量 ingest 在一个事务内提交，chunk_count 异步完成后才非 0"""
    factory = FakeUnitOfWorkFactory()
    handler = IngestBlobsCommandHandler(
        factory,
        RecursiveChunker(chunk_size=6000, chunk_overlap=200),
        FakeEmbedder(),
        FakeSearchStore(),
    )
    blobs = tuple(
        IngestBlobCommand(
            blob_name(f"src/{index}.py", f"print({index})"),
            f"src/{index}.py",
            f"print({index})",
        )
        for index in range(3)
    )

    result = await handler.handle(IngestBlobsCommand(blobs))

    # 异步模式：ingest 只写元数据，返回 0
    assert result.chunk_count == 0
    assert factory.uow.commits == 1
    # 验证 staging 已保存
    for blob in blobs:
        assert factory.uow.blobs.staging.get(blob.blob_name) == blob.content
