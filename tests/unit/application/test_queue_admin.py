"""队列重置命令的行为测试（用内存假件，不连 Redis）。"""

import pytest

from oce.application.commands.queue_admin import (
    ResetQueueCommand,
    ResetQueueCommandHandler,
)


class FakeQueue:
    """按 RedisQueue 的语义建模：主队列 + 处理中 + 在飞哨兵。"""

    def __init__(self, main: list[str], processing: list[str] | None = None) -> None:
        self.main = list(main)
        self.processing = list(processing or [])
        self.pending = set(self.main) | set(self.processing)

    async def enqueue(self, blob_name: str) -> None:
        if blob_name not in self.pending:
            self.pending.add(blob_name)
            self.main.insert(0, blob_name)

    async def size(self) -> int:
        return len(self.main)

    async def inflight_set(self) -> set[str]:
        return set(self.pending)

    async def purge(self) -> int:
        removed = len(self.main) + len(self.processing)
        self.main.clear()
        self.processing.clear()
        self.pending.clear()
        return removed

    async def retain(self, blob_names: set[str]) -> int:
        kept_main = [item for item in self.main if item in blob_names]
        kept_proc = [item for item in self.processing if item in blob_names]
        removed = (len(self.main) - len(kept_main)) + (
            len(self.processing) - len(kept_proc)
        )
        self.main = kept_main
        self.processing = kept_proc
        self.pending = set(kept_main) | set(kept_proc)
        return removed


class FakeBlobRepo:
    def __init__(self, pending: list[str]) -> None:
        self._pending = pending

    async def list_pending_names(self) -> list[str]:
        return list(self._pending)


class FakeUow:
    def __init__(self, pending: list[str]) -> None:
        self.blobs = FakeBlobRepo(pending)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def commit(self) -> None:
        pass


def make_handler(queue, pending: list[str]) -> ResetQueueCommandHandler:
    return ResetQueueCommandHandler(lambda: FakeUow(pending), queue)


@pytest.mark.asyncio
async def test_sync_drops_entries_missing_from_db():
    queue = FakeQueue(["ghost1", "live1", "ghost2", "live2"])
    handler = make_handler(queue, ["live1", "live2"])

    result = await handler.handle(ResetQueueCommand())

    assert result.removed == 2
    assert result.requeued == 0
    assert result.queue_size == 2
    assert set(queue.main) == {"live1", "live2"}
    # 哨兵必须跟着收缩，否则被剔除的名字再也无法入队
    assert queue.pending == {"live1", "live2"}


@pytest.mark.asyncio
async def test_sync_requeues_pending_blobs_absent_from_queue():
    queue = FakeQueue(["live1"])
    handler = make_handler(queue, ["live1", "live2", "live3"])

    result = await handler.handle(ResetQueueCommand())

    assert result.removed == 0
    assert result.requeued == 2
    assert result.queue_size == 3
    assert queue.pending == {"live1", "live2", "live3"}


@pytest.mark.asyncio
async def test_sync_clears_stale_sentinel_so_blob_can_requeue():
    """哨兵有残留但队列没有该条目时，同步后该 blob 必须能重新入队。"""
    queue = FakeQueue(["live1"])
    queue.pending.add("orphan-sentinel")
    handler = make_handler(queue, ["live1", "orphan-sentinel"])

    result = await handler.handle(ResetQueueCommand())

    assert result.requeued == 1
    assert "orphan-sentinel" in queue.main


@pytest.mark.asyncio
async def test_purge_then_requeues_everything_from_db():
    queue = FakeQueue(["ghost1", "ghost2"], processing=["ghost3"])
    handler = make_handler(queue, ["live1", "live2"])

    result = await handler.handle(ResetQueueCommand(mode="purge"))

    assert result.removed == 3
    assert result.requeued == 2
    assert set(queue.main) == {"live1", "live2"}


@pytest.mark.asyncio
async def test_no_requeue_only_cleans():
    queue = FakeQueue(["ghost1", "live1"])
    handler = make_handler(queue, ["live1", "live2"])

    result = await handler.handle(ResetQueueCommand(requeue=False))

    assert result.removed == 1
    assert result.requeued == 0
    assert queue.main == ["live1"]


@pytest.mark.asyncio
async def test_missing_queue_is_a_noop():
    handler = ResetQueueCommandHandler(lambda: FakeUow(["live1"]), None)

    result = await handler.handle(ResetQueueCommand())

    assert (result.removed, result.requeued, result.queue_size) == (0, 0, 0)
