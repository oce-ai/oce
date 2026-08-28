"""FindMissing / BlobStatus 对账查询处理器测试"""

from __future__ import annotations

import pytest

from oce.application.queries.status import (
    BlobStatusQuery,
    BlobStatusQueryHandler,
    FindMissingQuery,
    FindMissingQueryHandler,
    ResolveScopeQuery,
    ResolveScopeQueryHandler,
)
from oce.domain.blob.blob import Blob, BlobStatus
from oce.domain.chain.chain import Chain
from oce.shared.errors import (
    InvalidCheckpointTokenError,
    NeedsResetError,
    ScopeRequiredError,
)

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


class TestResolveScopeQueryHandler:
    """检索范围解析：全库检索已禁用，必须正面声明工作集。"""

    async def test_without_client_scope_raises(self, repos):
        factory, _, _ = repos
        with pytest.raises(ScopeRequiredError):
            await ResolveScopeQueryHandler(factory).handle(ResolveScopeQuery())

    async def test_deleted_without_positive_scope_raises(self, repos):
        # deleted_blobs 只是减法，不构成工作集声明
        factory, _, _ = repos
        with pytest.raises(ScopeRequiredError):
            await ResolveScopeQueryHandler(factory).handle(
                ResolveScopeQuery(deleted_blobs=("a",))
            )

    async def test_added_blobs_only_forms_scope(self, repos):
        factory, _, _ = repos
        result = await ResolveScopeQueryHandler(factory).handle(
            ResolveScopeQuery(added_blobs=("a", "b"), deleted_blobs=("b",))
        )
        assert result.blob_names == frozenset({"a"})

    async def test_malformed_token_raises_invalid(self, repos):
        factory, _, _ = repos
        with pytest.raises(InvalidCheckpointTokenError):
            await ResolveScopeQueryHandler(factory).handle(
                ResolveScopeQuery(checkpoint_id="bad-token")
            )

    async def test_missing_chain_raises_needs_reset(self, repos):
        factory, _, _ = repos
        ghost = f"{Chain.create(['x']).chain_id}:1"
        with pytest.raises(NeedsResetError):
            await ResolveScopeQueryHandler(factory).handle(
                ResolveScopeQuery(checkpoint_id=ghost)
            )

    async def test_checkpoint_members_plus_increments(self, repos):
        factory, _, chain_repo = repos
        chain = await chain_repo.create(["a", "b"])
        result = await ResolveScopeQueryHandler(factory).handle(
            ResolveScopeQuery(
                checkpoint_id=chain.get_checkpoint_token(),
                added_blobs=("c",),
                deleted_blobs=("b",),
            )
        )
        assert result.blob_names == frozenset({"a", "c"})

    async def test_empty_chain_is_empty_scope_not_error(self, repos):
        # 有效但成员为空的 checkpoint → 空工作集（空结果），不算全库检索
        factory, _, chain_repo = repos
        chain = await chain_repo.create([])
        result = await ResolveScopeQueryHandler(factory).handle(
            ResolveScopeQuery(checkpoint_id=chain.get_checkpoint_token())
        )
        assert result.blob_names == frozenset()
