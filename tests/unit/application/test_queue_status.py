"""QueueStatusQueryHandler 单元测试。"""

from __future__ import annotations

from oce.application.queries.queue import (
    QueueStatusQuery,
    QueueStatusQueryHandler,
)


class _FakeBlobs:
    async def list_pending_names(self):
        return ["a", "b", "c", "d"]


class _FakeUow:
    def __init__(self) -> None:
        self.blobs = _FakeBlobs()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


def _uow_factory():
    return _FakeUow()


class _FakeQueue:
    async def size(self):
        return 3

    async def inflight_set(self):
        return {"a", "b"}


async def test_queue_status_reports_counts():
    handler = QueueStatusQueryHandler(_uow_factory, _FakeQueue())
    result = await handler.handle(QueueStatusQuery())
    assert result.enabled is True
    assert result.main_size == 3
    assert result.inflight == 2
    assert result.db_pending == 4


async def test_queue_status_disabled_without_queue():
    handler = QueueStatusQueryHandler(_uow_factory, None)
    result = await handler.handle(QueueStatusQuery())
    assert result.enabled is False
    assert result.main_size == 0
    assert result.inflight == 0
    assert result.db_pending == 0
