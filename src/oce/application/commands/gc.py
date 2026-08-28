"""过期数据回收（GC）：dry-run 优先。

删除口径：
- 过期 chain：updated_at 早于 now - ttl_days，删除只移除 checkpoint 分组，不动 blob。
- 过期 blob：last_seen 早于 now - ttl_days 且不在 queue 的 inflight 集合内，经
  DeleteBlobsCommand 连带清 DB/向量/路径。inflight 项跳过，避免删正在嵌入的 blob。

dry_run=True（默认）只统计不删；dry_run=False 才真正执行删除。
"""

from __future__ import annotations

from dataclasses import dataclass

from oce.application.commands.ingest import (
    DeleteBlobsCommand,
    DeleteBlobsCommandHandler,
)
from oce.application.messages import Command
from oce.application.queue import Queue
from oce.application.uow import UnitOfWorkFactory


@dataclass(frozen=True)
class GcCommand(Command):
    ttl_days: int = 30
    dry_run: bool = True
    limit: int = 1000


@dataclass(frozen=True)
class GcResult:
    dry_run: bool
    ttl_days: int
    expired_chains: int
    expired_blobs: int
    deletable_blobs: int
    skipped_inflight: int
    deleted_chains: int
    deleted_blobs: int


class GcCommandHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        delete_blobs: DeleteBlobsCommandHandler,
        queue: Queue | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._delete_blobs = delete_blobs
        self._queue = queue

    async def handle(self, command: GcCommand) -> GcResult:
        async with self._uow_factory() as uow:
            expired_chains = list(await uow.chains.find_expired(command.ttl_days))
            expired_blobs = list(
                await uow.blobs.find_expired(command.ttl_days, batch_size=command.limit)
            )

        inflight: set[str] = (
            await self._queue.inflight_set() if self._queue is not None else set()
        )
        deletable = [name for name in expired_blobs if name not in inflight]
        skipped = len(expired_blobs) - len(deletable)

        if command.dry_run:
            return GcResult(
                dry_run=True,
                ttl_days=command.ttl_days,
                expired_chains=len(expired_chains),
                expired_blobs=len(expired_blobs),
                deletable_blobs=len(deletable),
                skipped_inflight=skipped,
                deleted_chains=0,
                deleted_blobs=0,
            )

        async with self._uow_factory() as uow:
            for chain_id in expired_chains:
                await uow.chains.delete(chain_id)
            await uow.commit()

        if deletable:
            await self._delete_blobs.handle(DeleteBlobsCommand(tuple(deletable)))

        return GcResult(
            dry_run=False,
            ttl_days=command.ttl_days,
            expired_chains=len(expired_chains),
            expired_blobs=len(expired_blobs),
            deletable_blobs=len(deletable),
            skipped_inflight=skipped,
            deleted_chains=len(expired_chains),
            deleted_blobs=len(deletable),
        )
