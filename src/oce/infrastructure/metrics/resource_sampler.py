"""资源采样器：后台周期采集磁盘 / 内存 / CPU 到 MetricsSink，落 resource_samples。

psutil 惰性导入：缺失时优雅降级（记一次日志、不采样），绝不拖垮启动。采样与写库都
走旁路，异常只记日志。collector 为 None（psutil 缺失或监控关闭）时 start() 直接跳过。
"""
from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Callable

from loguru import logger

from oce.shared.metrics import MetricsSink, ResourceSampleRecord

ResourceCollector = Callable[[], ResourceSampleRecord]


def _dir_size(path: str) -> int:
    """递归累加目录内文件字节数；单个文件不可读则跳过，整体不可达返回 0。"""
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
    except OSError:
        return 0
    return total


def build_psutil_collector(data_dir: str | None) -> ResourceCollector | None:
    """构造基于 psutil 的采集器；psutil 不可用时返回 None（调用方据此禁用采样）。"""
    try:
        import psutil
    except ImportError:
        logger.warning("psutil not installed; resource sampling disabled")
        return None

    proc = psutil.Process()
    proc.cpu_percent(None)  # 预热基线：首个 interval 的 CPU% 才有意义
    target = data_dir or os.getcwd()

    def _collect() -> ResourceSampleRecord:
        usage = shutil.disk_usage(target)
        return ResourceSampleRecord(
            disk_data_bytes=_dir_size(data_dir) if data_dir else 0,
            disk_free_bytes=usage.free,
            disk_total_bytes=usage.total,
            mem_rss_bytes=proc.memory_info().rss,
            mem_percent=float(proc.memory_percent()),
            cpu_percent=float(proc.cpu_percent(None)),
        )

    return _collect


class ResourceSampler:
    """后台周期采样。个人 / 服务模式都跑；collector 为 None 时整体禁用。"""

    def __init__(
        self,
        sink: MetricsSink,
        *,
        interval_seconds: float,
        collector: ResourceCollector | None,
    ) -> None:
        self._sink = sink
        self._interval = interval_seconds
        self._collector = collector
        self._task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        if self._running or self._collector is None:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                self._tick()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("resource sample loop error: {}", exc)

    def _tick(self) -> None:
        if self._collector is None:
            return
        try:
            self._sink.record_resource_sample(self._collector())
        except Exception as exc:  # 旁路容错：采样失败不影响主进程
            logger.warning("resource sample failed: {}", exc)
