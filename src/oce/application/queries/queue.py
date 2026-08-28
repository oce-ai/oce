"""队列健康查询：主队列长度 / 在飞数 / DB 待办数。

queue 为 None（个人模式或 worker 关闭）时返回 enabled=False 的零值快照。
"""

from __future__ import annotations

from dataclasses import dataclass

from oce.application.messages import Query
from oce.application.queue import Queue
from oce.application.uow import UnitOfWorkFactory


@dataclass(frozen=True)
class QueueStatusQuery(Query):
    pass


@dataclass(frozen=True)
class QueueStatusResult:
    enabled: bool
    main_size: int
    inflight: int
    db_pending: int


class QueueStatusQueryHandler:
    def __init__(
        self,
        uow_factory: UnitOfWorkFactory,
        queue: Queue | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._queue = queue

    async def handle(self, _query: QueueStatusQuery) -> QueueStatusResult:
        if self._queue is None:
            return QueueStatusResult(
                enabled=False, main_size=0, inflight=0, db_pending=0
            )
        async with self._uow_factory() as uow:
            db_pending = len(await uow.blobs.list_pending_names())
        return QueueStatusResult(
            enabled=True,
            main_size=await self._queue.size(),
            inflight=len(await self._queue.inflight_set()),
            db_pending=db_pending,
        )
