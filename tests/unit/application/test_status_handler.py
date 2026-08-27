"""FindMissing / BlobStatus 对账查询处理器测试"""

from __future__ import annotations

import pytest

from oce.application.queries.status import (
    BlobStatusQuery,
    BlobStatusQueryHandler,
    FindMissingQuery,
    FindMissingQueryHandler,
    OverviewContextQuery,
    OverviewContextQueryHandler,
)
from oce.domain.blob.blob import Blob, BlobStatus
from oce.domain.chain.chain import Chain

from tests.unit.application.fakes import FakeUnitOfWorkFactory, blob_name


@pytest.fixture
def repos():
    factory = FakeUnitOfWorkFactory()
    return factory, factory.uow.blobs, factory.uow.chains


async def _save_ready_blob(blob_repo, path: str, content: str) -> str:
    """存一个 ready 的 blob"""
    name = blob_name(path, content)
    blob = Blob(blob_name=name, path=path, status=BlobStatus.READY)
    blob_repo.blobs[name] = blob
    return name


class TestFindMissingQueryHandler:
    async def test_classifies_unknown_and_nonindexed(self, repos):
        factory, blob_repo, _ = repos
        ready = await _save_ready_blob(blob_repo, "src/a.py", "print(1)\n")
        pending = blob_name("src/b.py", "print(2)\n")
        blob_repo.blobs[pending] = Blob(blob_name=pending, path="src/b.py")

        handler = FindMissingQueryHandler(factory)
        result = await handler.handle(
            FindMissingQuery(blob_names=(ready, pending, "ghost"))
        )

        assert result.unknown == ("ghost",)
        assert result.nonindexed == (pending,)

    async def test_empty_query_returns_empty(self, repos):
        factory, blob_repo, _ = repos
        result = await FindMissingQueryHandler(factory).handle(
            FindMissingQuery(blob_names=())
        )
        assert result.unknown == ()
        assert result.nonindexed == ()


class TestBlobStatusQueryHandler:
    async def test_no_checkpoint_id_is_noop(self, repos):
        factory, blob_repo, chain_repo = repos
        result = await BlobStatusQueryHandler(factory).handle(
            BlobStatusQuery(blob_names=())
        )
        assert result.checkpoint_not_found is False

    async def test_malformed_token_reports_not_found(self, repos):
        factory, _, _ = repos
        result = await BlobStatusQueryHandler(factory).handle(
            BlobStatusQuery(checkpoint_id="bad-token")
        )
        assert result.checkpoint_not_found is True

    async def test_valid_token_checks_chain(self, repos):
        factory, _, chain_repo = repos
        chain = await chain_repo.create(["a"])
        handler = BlobStatusQueryHandler(factory)

        found = await handler.handle(
            BlobStatusQuery(checkpoint_id=chain.get_checkpoint_token())
        )
        missing = await handler.handle(
            BlobStatusQuery(checkpoint_id=f"{Chain.create(['x']).chain_id}:1")
        )

        assert found.checkpoint_not_found is False
        assert missing.checkpoint_not_found is True

    async def test_combines_blob_status(self, repos):
        factory, blob_repo, chain_repo = repos
        ready = await _save_ready_blob(blob_repo, "src/a.py", "print(1)\n")
        chain = await chain_repo.create(["a"])

        result = await BlobStatusQueryHandler(factory).handle(
            BlobStatusQuery(
                blob_names=(ready, "ghost"),
                checkpoint_id=chain.get_checkpoint_token(),
            )
        )

        assert result.unknown == ("ghost",)
        assert result.nonindexed == ()
        assert result.checkpoint_not_found is False


async def test_overview_context_returns_key_docs_and_bounded_paths(repos):
    factory, blob_repo, _ = repos
    readme = await _save_ready_blob(blob_repo, "README.md", "# Project\nDetails")
    source = await _save_ready_blob(blob_repo, "src/main.py", "print('hi')")
    blob_repo.contents[readme] = ("README.md", "# Project\nDetails")

    result = await OverviewContextQueryHandler(factory).handle(
        OverviewContextQuery(
            frozenset({readme, source}),
            paths_limit=1,
            key_doc_max_bytes=12,
        )
    )

    assert result.paths == ("README.md",)
    assert result.paths_total == 2
    assert len(result.key_docs) == 1
    assert result.key_docs[0].path == "README.md"
    assert result.key_docs[0].content == "# Project"
    assert result.key_docs[0].truncated is True


async def test_resolve_scope_without_client_scope_returns_none(repos):
    from oce.application.queries.status import ResolveScopeQuery, ResolveScopeQueryHandler

    factory, _, _ = repos
    result = await ResolveScopeQueryHandler(factory).handle(ResolveScopeQuery())

    # 无 checkpoint、无 added blobs = 客户端未给范围 → None（不过滤，检索全库）；
    # 空 frozenset 会被 pipeline 当成"无可搜内容"直接返回 []，语义相反。
    assert result.blob_names is None
