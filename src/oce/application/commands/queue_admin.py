"""队列运维命令：对齐队列与 DB 待办，或整体清空。

队列是 DB 待办的投影，不是权威。两者会漂移：blob 被删除或重建后旧消息仍在飞，
worker 取到只能白跑；更糟的是 pending 哨兵里的残留会让同名 blob 再也无法入队。

`ResetQueueCommand` 提供两档处置：
- mode="sync"（默认）：以 DB 的 pending blob 为准剔除队列里的无效项，并补投漏投项
- mode="purge"：清空队列再按 DB 全量重投

两者都要求 worker 已停；handler 不会替调用方停 worker，因为 worker 生命周期
由 composition root 持有。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from oce.application.messages import Command
from oce.application.queue import Queue
from oce.application.uow import UnitOfWorkFactory


@dataclass(frozen=True)
class ResetQueueCommand(Command):
    """重置队列，使其与 DB 的 pending blob 一致。

    requeue=False 时只做清理不投递，用于「先停下来看看」的场景。
    """

    mode: Literal["sync", "purge"] = "sync"
    requeue: bool = True


@dataclass(frozen=True)
class ResetQueueResult:
    """removed 为剔除条数，requeued 为本次投递条数，queue_size 为结束时主队列长度。"""

    removed: int
    requeued: int
    queue_size: int
    db_pending: int


class ResetQueueCommandHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        queue: Queue | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._queue = queue

    async def handle(self, command: ResetQueueCommand) -> ResetQueueResult:
        if self._queue is None:
            return ResetQueueResult(0, 0, 0, 0)

        async with self._uow_factory() as uow:
            pending = set(await uow.blobs.list_pending_names())

        if command.mode == "purge":
            removed = await self._queue.purge()
        else:
            removed = await self._queue.retain(pending)

        requeued = 0
        if command.requeue:
            inflight = await self._queue.inflight_set()
            for blob_name in sorted(pending - inflight):
                await self._queue.enqueue(blob_name)
                requeued += 1

        return ResetQueueResult(
            removed=removed,
            requeued=requeued,
            queue_size=await self._queue.size(),
            db_pending=len(pending),
        )
