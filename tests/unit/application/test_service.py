"""Tests for cross-handler application orchestration."""

from __future__ import annotations

from types import SimpleNamespace

from oce.application.commands.checkpoint import CheckpointCommand
from oce.application.commands.ingest import (
    DeleteBlobsCommand,
    EmbedPendingCommand,
    IngestBlobsCommand,
)
from oce.application.queries.search import SearchQuery
from oce.application.queries.status import (
    ResolveScopeQuery,
    ResolveScopeResult,
)
from oce.application.service import BlobUpload, RetrievalApplication, compute_blob_name
from oce.domain.services.search import SearchHit


class SpyCommandBus:
    def __init__(self) -> None:
        self.commands: list[object] = []

    async def execute(self, command):
        self.commands.append(command)
        if isinstance(command, EmbedPendingCommand):
            return SimpleNamespace(embedded_count=0)
        return None


class SpyQueryBus:
    def __init__(self) -> None:
        self.queries: list[object] = []

    async def ask(self, query):
        self.queries.append(query)
        if isinstance(query, ResolveScopeQuery):
            return ResolveScopeResult(frozenset({"blob-a"}))
        if isinstance(query, SearchQuery):
            return SimpleNamespace(
                hits=[
                    SearchHit(
                        blob_name="blob-a",
                        path="src/a.py",
                        content="def a(): pass",
                        score=0.9,
                    )
                ]
            )
        raise AssertionError(f"Unexpected query: {query!r}")


async def test_retrieve_passes_deleted_blobs_to_scope_without_delete_side_effect():
    commands = SpyCommandBus()
    queries = SpyQueryBus()
    application = RetrievalApplication(commands, queries)

    result = await application.retrieve(
        "entry point",
        checkpoint_id="chain:1",
        added_blobs=["blob-a"],
        deleted_blobs=["blob-b"],
    )

    assert not any(isinstance(item, DeleteBlobsCommand) for item in commands.commands)
    scope_query = next(q for q in queries.queries if isinstance(q, ResolveScopeQuery))
    assert scope_query.checkpoint_id == "chain:1"
    assert scope_query.added_blobs == ("blob-a",)
    assert scope_query.deleted_blobs == ("blob-b",)
    assert result.hits


async def test_retrieve_with_empty_scope_requests_empty_search():
    class EmptyScopeQueries(SpyQueryBus):
        async def ask(self, query):
            self.queries.append(query)
            if isinstance(query, ResolveScopeQuery):
                return ResolveScopeResult(frozenset())
            if isinstance(query, SearchQuery):
                assert query.allowed_blob_names == frozenset()
                return SimpleNamespace(hits=[])
            raise AssertionError(f"Unexpected query: {query!r}")

    application = RetrievalApplication(SpyCommandBus(), EmptyScopeQueries())

    result = await application.retrieve("entry point")

    assert result.hits == ()


async def test_batch_upload_does_not_embed_synchronously_with_background_worker():
    class BatchCommandBus(SpyCommandBus):
        async def execute(self, command):
            self.commands.append(command)
            if isinstance(command, IngestBlobsCommand):
                return SimpleNamespace(chunk_count=0)
            if isinstance(command, EmbedPendingCommand):
                raise AssertionError("background mode must not embed synchronously")
            raise AssertionError(f"Unexpected command: {command!r}")

    commands = BatchCommandBus()
    application = RetrievalApplication(
        commands,
        SpyQueryBus(),
        background_indexing=True,
    )

    result = await application.batch_upload([])

    assert result.embedded_count == 0
    assert sum(isinstance(item, IngestBlobsCommand) for item in commands.commands) == 1
    assert not any(isinstance(item, EmbedPendingCommand) for item in commands.commands)


class UploadCommandBus(SpyCommandBus):
    """完整响应 batch_upload 流程的 command bus（ingest → embed → checkpoint）"""

    async def execute(self, command):
        self.commands.append(command)
        if isinstance(command, IngestBlobsCommand):
            return SimpleNamespace(chunk_count=0)
        if isinstance(command, EmbedPendingCommand):
            return SimpleNamespace(embedded_count=0)
        if isinstance(command, CheckpointCommand):
            return SimpleNamespace(new_checkpoint_id=f"{command.checkpoint_id}_v1")
        return None


async def test_batch_upload_registers_blobs_to_checkpoint_when_id_given():
    commands = UploadCommandBus()
    application = RetrievalApplication(commands, SpyQueryBus())
    blob = BlobUpload("src/a.py", "print(1)\n")

    await application.batch_upload([blob], checkpoint_id="chain:2")

    checkpoint = next(c for c in commands.commands if isinstance(c, CheckpointCommand))
    assert checkpoint.checkpoint_id == "chain:2"
    assert checkpoint.added_blobs == (compute_blob_name(blob.path, blob.content),)
    assert checkpoint.deleted_blobs == ()


async def test_batch_upload_without_checkpoint_id_skips_checkpoint():
    commands = UploadCommandBus()
    application = RetrievalApplication(commands, SpyQueryBus())

    await application.batch_upload([BlobUpload("src/a.py", "print(1)\n")])

    assert not any(isinstance(c, CheckpointCommand) for c in commands.commands)
