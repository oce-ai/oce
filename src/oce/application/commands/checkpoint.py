"""Checkpoint 工作集命令。"""

from __future__ import annotations

from dataclasses import dataclass

from oce.application.messages import Command
from oce.application.uow import UnitOfWorkFactory
from oce.domain.chain.chain import Chain
from oce.shared.errors import InvalidCheckpointTokenError, NeedsResetError


@dataclass(frozen=True)
class CheckpointCommand(Command):
    checkpoint_id: str | None = None
    added_blobs: tuple[str, ...] = ()
    deleted_blobs: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckpointResult:
    new_checkpoint_id: str


class CheckpointCommandHandler:
    def __init__(self, uow_factory: UnitOfWorkFactory) -> None:
        self._uow_factory = uow_factory

    async def handle(self, command: CheckpointCommand) -> CheckpointResult:
        async with self._uow_factory() as uow:
            if not command.checkpoint_id:
                members = sorted(set(command.added_blobs) - set(command.deleted_blobs))
                chain = await uow.chains.create(members)
                chain_id, version = chain.chain_id, chain.version
            else:
                parsed = Chain.parse_checkpoint_token(command.checkpoint_id)
                if parsed is None:
                    raise InvalidCheckpointTokenError(command.checkpoint_id)
                chain_id, _ = parsed
                version = await uow.chains.apply_checkpoint(
                    chain_id,
                    command.added_blobs,
                    command.deleted_blobs,
                )
                if version is None:
                    raise NeedsResetError("checkpoint 链不存在（服务端状态丢失）")
            await uow.chains.touch_members(chain_id)
            await uow.commit()
        return CheckpointResult(f"{chain_id}:{version}")
