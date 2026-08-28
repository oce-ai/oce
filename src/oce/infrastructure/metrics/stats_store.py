"""监控只读聚合的 SQL 实现（跨 SQLite / PostgreSQL 可移植）。

避免依赖方言专有的 percentile 函数：延迟分位在 Python 侧对窗口内延迟排序后计算。
监控是旁路，本读路径只读不写，不影响主链路。
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from oce.infrastructure.persistence.models import (
    ApiCallMetricModel,
    ResourceSampleModel,
    RetrievalMetricModel,
    TokenUsageMetricModel,
)
from oce.shared.metrics_read import (
    ApiCallStats,
    MonitoringStats,
    ResourceSnapshot,
    RetrievalStats,
    TokenKindStats,
)


def _percentile(sorted_vals: list[int], p: int) -> int:
    if not sorted_vals:
        return 0
    idx = int(round((p / 100) * (len(sorted_vals) - 1)))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


def _api_stats(rows: list[tuple[int, int]]) -> ApiCallStats:
    if not rows:
        return ApiCallStats()
    latencies = sorted(int(latency) for latency, _status in rows)
    errors = sum(1 for _latency, status in rows if status >= 500)
    count = len(latencies)
    return ApiCallStats(
        count=count,
        error_count=errors,
        avg_latency_ms=round(sum(latencies) / count, 2),
        p50_latency_ms=_percentile(latencies, 50),
        p95_latency_ms=_percentile(latencies, 95),
        max_latency_ms=latencies[-1],
    )


def _snapshot(row) -> ResourceSnapshot | None:
    if row is None:
        return None
    return ResourceSnapshot(
        ts=row.ts,
        mem_rss_bytes=row.mem_rss_bytes,
        mem_percent=row.mem_percent,
        cpu_percent=row.cpu_percent,
        disk_free_bytes=row.disk_free_bytes,
        disk_total_bytes=row.disk_total_bytes,
        disk_data_bytes=row.disk_data_bytes,
    )


class SqlMonitoringStatsReader:
    """按时间窗口聚合监控四表；延迟分位在 Python 侧算，跨方言可移植。"""

    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def read(self, window_hours: int) -> MonitoringStats:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
        async with self._session_factory() as session:
            api_rows = (
                await session.execute(
                    select(
                        ApiCallMetricModel.latency_ms,
                        ApiCallMetricModel.status_code,
                    ).where(ApiCallMetricModel.ts >= cutoff)
                )
            ).all()

            token_rows = (
                await session.execute(
                    select(
                        TokenUsageMetricModel.kind,
                        func.count(),
                        func.coalesce(func.sum(TokenUsageMetricModel.prompt_tokens), 0),
                        func.coalesce(func.sum(TokenUsageMetricModel.completion_tokens), 0),
                        func.coalesce(func.sum(TokenUsageMetricModel.total_tokens), 0),
                    )
                    .where(TokenUsageMetricModel.ts >= cutoff)
                    .group_by(TokenUsageMetricModel.kind)
                )
            ).all()

            empty_expr = func.coalesce(
                func.sum(case((RetrievalMetricModel.hit_count == 0, 1), else_=0)), 0
            )
            retrieval_row = (
                await session.execute(
                    select(func.count(), empty_expr).where(
                        RetrievalMetricModel.ts >= cutoff
                    )
                )
            ).one()

            resource_row = (
                await session.execute(
                    select(ResourceSampleModel)
                    .order_by(ResourceSampleModel.ts.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()

        tokens = tuple(
            TokenKindStats(
                kind=kind,
                calls=int(calls),
                prompt_tokens=int(prompt),
                completion_tokens=int(completion),
                total_tokens=int(total),
            )
            for kind, calls, prompt, completion, total in token_rows
        )
        r_count = int(retrieval_row[0] or 0)
        r_empty = int(retrieval_row[1] or 0)
        retrieval = RetrievalStats(
            count=r_count,
            empty_count=r_empty,
            empty_rate=round(r_empty / r_count, 4) if r_count else 0.0,
        )
        return MonitoringStats(
            window_hours=window_hours,
            api_calls=_api_stats(api_rows),
            tokens=tokens,
            tokens_total=sum(t.total_tokens for t in tokens),
            retrieval=retrieval,
            resource=_snapshot(resource_row),
        )
