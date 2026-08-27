"""重新入队卡住的 blob,让重试生效。"""

from __future__ import annotations

from dataclasses import dataclass

from oce.application.messages import Command
from oce.application.uow import UnitOfWorkFactory


@dataclass(frozen=True)
class RequeueStaleCommand(Command):
    """重新入队长时间未处理的 pending blob"""
    stale_hours: int = 24
    limit: int = 100


@dataclass(frozen=True)
class RequeueStaleResult:
    requeued_count: int


class RequeueStaleCommandHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        queue=None,  # 可选:启用异步时传入 RedisQueue
    ) -> None:
        self._uow_factory = uow_factory
        self._queue = queue

    async def handle(self, command: RequeueStaleCommand) -> RequeueStaleResult:
        if self._queue is None:
            return RequeueStaleResult(0)

        async with self._uow_factory() as uow:
            stale = await uow.blobs.find_stale_with_staging(
                stale_hours=command.stale_hours,
                limit=command.limit,
            )
            for blob_name in stale:
                await self._queue.enqueue(blob_name)
            await uow.commit()

        return RequeueStaleResult(len(stale))
