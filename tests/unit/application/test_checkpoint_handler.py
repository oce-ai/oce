"""CheckpointCommand 处理器测试"""

from __future__ import annotations

import pytest

from oce.application.commands.checkpoint import (
    CheckpointCommand,
    CheckpointCommandHandler,
)
from oce.domain.chain.chain import Chain
from oce.shared.errors import InvalidCheckpointTokenError, NeedsResetError

from tests.unit.application.fakes import FakeUnitOfWorkFactory


@pytest.fixture
def handler():
    return CheckpointCommandHandler(FakeUnitOfWorkFactory())


class TestCheckpointCommandHandler:
    async def test_create_new_chain_returns_version_1(self, handler):
        result = await handler.handle(
            CheckpointCommand(added_blobs=("a", "b", "c"))
        )

        assert result.new_checkpoint_id.endswith(":1")
        chain_id = result.new_checkpoint_id.rsplit(":", 1)[0]
        members = await handler._uow_factory.uow.chains.get_members(chain_id)
        assert members == {"a", "b", "c"}

    async def test_create_new_chain_subtracts_deleted(self, handler):
        result = await handler.handle(
            CheckpointCommand(added_blobs=("a", "b"), deleted_blobs=("b",))
        )

        chain_id = result.new_checkpoint_id.rsplit(":", 1)[0]
        members = await handler._uow_factory.uow.chains.get_members(chain_id)
        assert members == {"a"}

    async def test_apply_incremental_checkpoint(self, handler):
        first = await handler.handle(CheckpointCommand(added_blobs=("a",)))

        result = await handler.handle(
            CheckpointCommand(
                checkpoint_id=first.new_checkpoint_id,
                added_blobs=("b",),
                deleted_blobs=("a",),
            )
        )

        assert result.new_checkpoint_id.endswith(":2")
        chain_id = result.new_checkpoint_id.rsplit(":", 1)[0]
        members = await handler._uow_factory.uow.chains.get_members(chain_id)
        assert members == {"b"}

    async def test_chain_missing_raises_needs_reset(self, handler):
        token = f"{Chain.create(['x']).chain_id}:1"  # 链不在 repo 里

        with pytest.raises(NeedsResetError):
            await handler.handle(
                CheckpointCommand(checkpoint_id=token, added_blobs=("a",))
            )

    async def test_malformed_token_raises_invalid(self, handler):
        with pytest.raises(InvalidCheckpointTokenError):
            await handler.handle(
                CheckpointCommand(checkpoint_id="bad-token", added_blobs=("a",))
            )
