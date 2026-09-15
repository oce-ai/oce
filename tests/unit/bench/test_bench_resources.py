"""进程内存采样测试（psutil 取代 ctypes，Commit 4 三处修正之一）。

锁两件事：① 周期采样当前 RSS、``peak_rss_mb()`` 取**观测最大值**（跨平台一致口径，
取代原 Windows 取当前值 / POSIX 取峰值的不一致）；② psutil 不可用时优雅降级为 0.0，
评测仍进行。用注入式 _proc，不依赖真实内存波动。
"""

from __future__ import annotations

import pytest

from oce.bench.resources import ResourceMeter, build_meter


class _FakeMem:
    def __init__(self, rss: int) -> None:
        self.rss = rss


class _FakeProc:
    """可编程假进程：每次 memory_info() 弹出 rss 序列里的下一个值。"""

    def __init__(self, rss_sequence: list[int]) -> None:
        self._sequence = list(rss_sequence)
        self.calls = 0

    def memory_info(self) -> _FakeMem:
        self.calls += 1
        if not self._sequence:
            raise RuntimeError("sequence exhausted")
        return _FakeMem(self._sequence.pop(0))


def _meter_with(sequence: list[int]) -> ResourceMeter:
    """构造 ResourceMeter 后把 _proc 换成可编程假进程（绕过真实 psutil）。"""
    meter = ResourceMeter()
    meter._proc = _FakeProc(sequence)
    # ResourceMeter 构造时已打一点（用真 proc），重置计数与峰值以隔离假进程
    meter._peak_bytes = 0
    meter._samples = 0
    return meter


class TestPeakSampling:
    def test_peak_is_observed_maximum(self):
        meter = _meter_with([100, 500, 300, 400])
        for _ in range(4):
            meter.sample()
        # 观测最大值 500 字节 -> MB
        assert meter.peak_rss_mb() == pytest.approx(500 / (1024 * 1024))
        assert meter.sample_count == 4

    def test_monotonic_growth_tracks_latest(self):
        meter = _meter_with([1_048_576, 2_097_152, 4_194_304])
        for _ in range(3):
            meter.sample()
        assert meter.peak_rss_mb() == pytest.approx(4.0)

    def test_declining_rss_keeps_peak(self):
        """RSS 回落不降峰值（峰值语义，非当前值）。"""
        meter = _meter_with([3_145_728, 1_048_576, 2_097_152])
        for _ in range(3):
            meter.sample()
        assert meter.peak_rss_mb() == pytest.approx(3.0)

    def test_sample_error_ignored_does_not_break_peak(self):
        """单次采样抛错（进程退出/权限）静默跳过，不污染峰值。"""
        meter = _meter_with([2_097_152])
        meter.sample()  # 2.0 MB
        meter._proc._sequence = []  # 后续 memory_info() 抛 RuntimeError
        meter.sample()  # 应被吞掉，不改峰值/计数
        assert meter.peak_rss_mb() == pytest.approx(2.0)
        assert meter.sample_count == 1


class TestDegradation:
    def test_no_psutil_yields_zero(self):
        """psutil 不可用（_proc=None）：peak 恒为 0.0，sample 静默 no-op。"""
        meter = ResourceMeter()
        meter._proc = None
        meter._peak_bytes = 0
        meter._samples = 0
        meter.sample()
        meter.sample()
        assert meter.peak_rss_mb() == 0.0
        assert meter.sample_count == 0


class TestFactory:
    def test_build_meter_returns_usable_instance(self):
        """工厂产出真实 ResourceMeter（有 psutil 时构造即打基线点）。"""
        meter = build_meter()
        assert isinstance(meter, ResourceMeter)
        # 真实 psutil 在场 -> 构造时已采一个基线点且 RSS>0
        if meter._proc is not None:
            assert meter.sample_count >= 1
            assert meter.peak_rss_mb() > 0.0
