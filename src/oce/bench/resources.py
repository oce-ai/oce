"""进程内存采样（psutil）。

替代 harness 里 Windows 专用的 ctypes ``psapi.GetProcessMemoryInfo`` 那段。改用 oce 已依赖
的 psutil，并修正原实现的一处不一致：

- 原 Windows 分支声明了 ``PROCESS_MEMORY_COUNTERS``（含 ``PeakWorkingSetSize``）却返回
  ``WorkingSetSize``（**当前**值）；POSIX 分支返回 ``ru_maxrss``（**峰值**）。两条路径口径
  不同，跨平台数字不可比。
- psutil 无跨平台的"进程峰值 RSS"API（Windows 侧拿不到 ru_maxrss 那种内核峰值）。故统一为
  **周期采样当前 RSS、客户端侧取最大**：跨平台一致、名副其实地反映"评测过程中的内存高点"。

采样在 harness 的上传/查询循环里被动触发（``sample()``），零额外线程。psutil 缺失时优雅
降级为 0.0（评测仍进行，只是不报内存）—— 与 resource_sampler 的惰性导入策略一致。
"""

from __future__ import annotations

from loguru import logger


class ResourceMeter:
    """评测期进程 RSS 采样器：``sample()`` 打点，``peak_rss_mb()`` 取观测最大值。"""

    def __init__(self) -> None:
        self._peak_bytes = 0
        self._samples = 0
        self._proc = None
        try:
            import psutil

            self._proc = psutil.Process()
        except Exception as exc:  # psutil 缺失或 Process() 失败：降级为不采样
            logger.warning("bench resource meter disabled: {}", exc)
            self._proc = None
        # 构造即打第一点，作为基线（进程刚起的常驻内存）
        self.sample()

    def sample(self) -> None:
        """采集一次当前 RSS 并更新峰值。psutil 不可用时静默跳过。"""
        if self._proc is None:
            return
        try:
            rss = self._proc.memory_info().rss
        except Exception:  # 进程已退出 / 权限问题：忽略本次采样
            return
        self._samples += 1
        if rss > self._peak_bytes:
            self._peak_bytes = rss

    def peak_rss_mb(self) -> float:
        """观测到的峰值 RSS（MB）；未采样（psutil 缺失）时返回 0.0。"""
        return self._peak_bytes / (1024 * 1024)

    @property
    def sample_count(self) -> int:
        return self._samples


def build_meter() -> ResourceMeter:
    """构造一个 ResourceMeter（工厂入口，便于测试替换）。"""
    return ResourceMeter()
