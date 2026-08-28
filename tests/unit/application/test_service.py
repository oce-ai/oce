"""Tests for cross-handler application orchestration."""

from __future__ import annotations

from types import SimpleNamespace

from oce.application.commands.ingest import (
    DeleteBlobsCommand,
    EmbedPendingCommand,
    IngestBlobsCommand,
)
from oce.application.queries.search import SearchQuery
from oce.application.queries.status import (
    OverviewContextQuery,
    OverviewContextResult,
    ResolveScopeQuery,
    ResolveScopeResult,
)
from oce.application.service import RetrievalApplication
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
        if isinstance(query, OverviewContextQuery):
            return OverviewContextResult((), ("src/a.py",), 1)
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


async def test_project_overview_prepares_scope_once():
    commands = SpyCommandBus()
    queries = SpyQueryBus()
    application = RetrievalApplication(commands, queries)

    result = await application.project_overview(
        depth="basic",
        checkpoint_id="chain:1",
        added_blobs=["blob-a"],
        deleted_blobs=["blob-b"],
    )

    # deleted_blobs 只移出 scope，禁止触发任何删除命令
    assert not any(isinstance(item, DeleteBlobsCommand) for item in commands.commands)
    assert sum(isinstance(item, EmbedPendingCommand) for item in commands.commands) == 1
    assert sum(isinstance(item, ResolveScopeQuery) for item in queries.queries) == 1
    assert sum(isinstance(item, OverviewContextQuery) for item in queries.queries) == 1
    assert sum(isinstance(item, SearchQuery) for item in queries.queries) == 4
    assert len(result.sections) == 4
    assert result.working_set_paths == ("src/a.py",)


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
