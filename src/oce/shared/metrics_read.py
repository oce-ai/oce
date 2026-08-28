"""监控只读聚合契约：/admin/stats 的读模型 DTO 与 reader 端口。

采集侧（metrics.py）负责写入；本模块只定义读出聚合的结果结构与 reader Protocol。
infra 实现 SQL 聚合、application 编排、api 映射 DTO——三层都只依赖这里的纯数据结构。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class ApiCallStats:
    count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: int = 0
    p95_latency_ms: int = 0
    max_latency_ms: int = 0


@dataclass(frozen=True)
class TokenKindStats:
    kind: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class RetrievalStats:
    count: int = 0
    empty_count: int = 0
    empty_rate: float = 0.0


@dataclass(frozen=True)
class ResourceSnapshot:
    ts: datetime | None = None
    mem_rss_bytes: int = 0
    mem_percent: float = 0.0
    cpu_percent: float = 0.0
    disk_free_bytes: int = 0
    disk_total_bytes: int = 0
    disk_data_bytes: int = 0


@dataclass(frozen=True)
class MonitoringStats:
    window_hours: int
    api_calls: ApiCallStats = field(default_factory=ApiCallStats)
    tokens: tuple[TokenKindStats, ...] = ()
    tokens_total: int = 0
    retrieval: RetrievalStats = field(default_factory=RetrievalStats)
    resource: ResourceSnapshot | None = None


class MonitoringStatsReader(Protocol):
    """监控聚合读端口；infra 用 SQL 实现，按时间窗口聚合监控四表。"""

    async def read(self, window_hours: int) -> MonitoringStats: ...
