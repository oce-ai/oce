"""监控统计查询：只读聚合监控数据供 /admin/stats 使用。

handler 只把查询转交注入的 reader（infra 的 SQL 实现），不含业务逻辑；
读模型与 reader 端口定义在 shared.metrics_read。
"""
from __future__ import annotations

from dataclasses import dataclass

from oce.application.messages import Query
from oce.shared.metrics_read import MonitoringStats, MonitoringStatsReader


@dataclass(frozen=True)
class MonitoringStatsQuery(Query):
    window_hours: int = 24


class MonitoringStatsQueryHandler:
    def __init__(self, reader: MonitoringStatsReader) -> None:
        self._reader = reader

    async def handle(self, query: MonitoringStatsQuery) -> MonitoringStats:
        return await self._reader.read(query.window_hours)
