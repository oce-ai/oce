"""SearchQuery 处理器测试"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from oce.application.queries.search import SearchQuery, SearchQueryHandler
from oce.domain.services.retrieval import RetrievalPipeline
from oce.domain.services.search import SearchHit

from tests.unit.application.fakes import FakeEmbedder, FakeSearchStore


@pytest.fixture
def handler():
    store = FakeSearchStore(hits=[
        SearchHit(
            blob_name="h1",
            path="src/main.py",
            content="def main(): pass",
            score=0.9,
            start_line=1,
            end_line=1,
        ),
    ])
    pipe = RetrievalPipeline(embedder=FakeEmbedder(), store=store)
    return SearchQueryHandler(pipe), store


class TestSearchQueryHandler:
    async def test_search_returns_hits(self, handler):
        search_handler, store = handler
        result = await search_handler.handle(
            SearchQuery(query="main entry", allowed_blob_names=frozenset({"h1"}))
        )

        assert len(result.hits) == 1
        assert result.hits[0].path == "src/main.py"

    async def test_search_passes_scope_to_store(self, handler):
        search_handler, store = handler
        await search_handler.handle(
            SearchQuery(query="q", allowed_blob_names=frozenset({"a", "b"}))
        )

        assert set(store.last_kwargs["allowed_blob_names"]) == {"a", "b"}

    async def test_search_without_scope_passes_none(self, handler):
        search_handler, store = handler
        await search_handler.handle(SearchQuery(query="q"))

        assert store.last_kwargs["allowed_blob_names"] is None


@pytest.mark.parametrize("fails", [False, True])
async def test_retirement_drains_old_requests_before_closing_resources(fails):
    class BlockingPipeline:
        started = asyncio.Event()
        release = asyncio.Event()

        async def search(self, *args):
            self.started.set()
            await self.release.wait()
            if fails:
                raise RuntimeError("search failed")
            return []

    pipeline = BlockingPipeline()
    close = AsyncMock()
    handler = SearchQueryHandler(pipeline, on_close=close)
    request = asyncio.create_task(handler.handle(SearchQuery("q")))
    await pipeline.started.wait()
    await handler.retire()
    close.assert_not_awaited()

    shutdown = asyncio.create_task(handler.aclose())
    await asyncio.sleep(0)
    assert not shutdown.done()
    pipeline.release.set()
    if fails:
        with pytest.raises(RuntimeError, match="search failed"):
            await request
    else:
        assert (await request).hits == []
    await shutdown
    await handler.aclose()
    close.assert_awaited_once()


async def test_cancelled_request_releases_retired_resources():
    started = asyncio.Event()

    class BlockingPipeline:
        async def search(self, *args):
            started.set()
            await asyncio.Event().wait()

    close = AsyncMock()
    handler = SearchQueryHandler(BlockingPipeline(), on_close=close)
    request = asyncio.create_task(handler.handle(SearchQuery("q")))
    await started.wait()
    await handler.retire()
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    await handler.aclose()
    close.assert_awaited_once()
