"""GcCommandHandler 单元测试：dry-run 计数 / 真删 / inflight 守卫。"""

from __future__ import annotations

from oce.application.commands.gc import GcCommand, GcCommandHandler


class _FakeChains:
    def __init__(self, expired):
        self._expired = expired
        self.deleted: list[str] = []

    async def find_expired(self, ttl_days):
        return list(self._expired)

    async def delete(self, chain_id):
        self.deleted.append(chain_id)


class _FakeBlobs:
    def __init__(self, expired):
        self._expired = expired

    async def find_expired(self, ttl_days, batch_size=1000):
        return list(self._expired)


class _FakeUow:
    def __init__(self, chains, blobs):
        self.chains = chains
        self.blobs = blobs
        self.committed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def commit(self):
        self.committed = True


def _uow_factory(chains, blobs):
    def factory():
        return _FakeUow(chains, blobs)

    return factory


class _FakeQueue:
    def __init__(self, inflight):
        self._inflight = inflight

    async def inflight_set(self):
        return set(self._inflight)


class _FakeDeleteHandler:
    def __init__(self):
        self.calls: list[tuple[str, ...]] = []

    async def handle(self, command):
        self.calls.append(command.blob_names)


async def test_gc_dry_run_counts_without_deleting():
    chains = _FakeChains(["c1", "c2"])
    blobs = _FakeBlobs(["b1", "b2", "b3"])
    deleter = _FakeDeleteHandler()
    handler = GcCommandHandler(_uow_factory(chains, blobs), deleter, _FakeQueue({"b3"}))

    result = await handler.handle(GcCommand(ttl_days=30, dry_run=True))

    assert result.dry_run is True
    assert result.expired_chains == 2
    assert result.expired_blobs == 3
    assert result.deletable_blobs == 2  # b3 在飞被跳过
    assert result.skipped_inflight == 1
    assert result.deleted_chains == 0
    assert result.deleted_blobs == 0
    assert chains.deleted == []
    assert deleter.calls == []


async def test_gc_real_delete_removes_chains_and_deletable_blobs():
    chains = _FakeChains(["c1", "c2"])
    blobs = _FakeBlobs(["b1", "b2", "b3"])
    deleter = _FakeDeleteHandler()
    handler = GcCommandHandler(_uow_factory(chains, blobs), deleter, _FakeQueue({"b3"}))

    result = await handler.handle(GcCommand(ttl_days=30, dry_run=False))

    assert result.dry_run is False
    assert result.deleted_chains == 2
    assert result.deleted_blobs == 2
    assert set(chains.deleted) == {"c1", "c2"}
    assert deleter.calls == [("b1", "b2")]


async def test_gc_without_queue_treats_all_expired_as_deletable():
    chains = _FakeChains([])
    blobs = _FakeBlobs(["b1", "b2"])
    deleter = _FakeDeleteHandler()
    handler = GcCommandHandler(_uow_factory(chains, blobs), deleter, queue=None)

    result = await handler.handle(GcCommand(dry_run=True))

    assert result.deletable_blobs == 2
    assert result.skipped_inflight == 0
