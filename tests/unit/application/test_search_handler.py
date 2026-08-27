"""SearchQuery 处理器测试"""

from __future__ import annotations

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
