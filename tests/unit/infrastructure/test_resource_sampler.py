"""ResourceSampler 单测：注入假采集器验证 tick/启停/禁用/容错，不依赖 psutil。"""
from __future__ import annotations

import asyncio

from oce.infrastructure.metrics.resource_sampler import (
    ResourceSampler,
    build_psutil_collector,
)
from oce.shared.metrics import ResourceSampleRecord


class _RecordingSink:
    def __init__(self) -> None:
        self.samples: list[ResourceSampleRecord] = []

    def record_resource_sample(self, record: ResourceSampleRecord) -> None:
        self.samples.append(record)


def _fake_record() -> ResourceSampleRecord:
    return ResourceSampleRecord(
        disk_data_bytes=1,
        disk_free_bytes=2,
        disk_total_bytes=3,
        mem_rss_bytes=4,
        mem_percent=5.0,
        cpu_percent=6.0,
    )


async def test_tick_records_one_sample():
    sink = _RecordingSink()
    sampler = ResourceSampler(sink, interval_seconds=999, collector=_fake_record)
    sampler._tick()
    assert len(sink.samples) == 1
    assert sink.samples[0].disk_total_bytes == 3


async def test_start_stop_runs_loop():
    sink = _RecordingSink()
    sampler = ResourceSampler(sink, interval_seconds=0.01, collector=_fake_record)
    await sampler.start()
    await asyncio.sleep(0.05)
    await sampler.stop()
    assert len(sink.samples) >= 1


async def test_none_collector_disables_sampling():
    sink = _RecordingSink()
    sampler = ResourceSampler(sink, interval_seconds=0.01, collector=None)
    await sampler.start()  # 无采集器：直接跳过，不建 task
    await asyncio.sleep(0.02)
    await sampler.stop()
    assert sink.samples == []


async def test_tick_swallows_collector_error():
    sink = _RecordingSink()

    def _boom() -> ResourceSampleRecord:
        raise RuntimeError("x")

    sampler = ResourceSampler(sink, interval_seconds=999, collector=_boom)
    sampler._tick()  # 旁路容错：不抛即通过
    assert sink.samples == []


def test_build_psutil_collector_shape(tmp_path):
    """psutil 可用则采出真实快照；不可用则优雅降级为 None。"""
    collector = build_psutil_collector(str(tmp_path))
    if collector is None:
        return  # psutil 未安装：降级路径已生效
    record = collector()
    assert record.disk_total_bytes > 0
    assert record.mem_rss_bytes > 0
    assert record.cpu_percent >= 0.0
