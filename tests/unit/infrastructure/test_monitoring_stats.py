"""SqlMonitoringStatsReader 单测：窗口内聚合 + 分位/空回率/最新资源快照，窗口外排除。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import oce.infrastructure.persistence.models  # noqa: F401  注册 ORM 表到 Base.metadata
from oce.infrastructure.metrics.stats_store import SqlMonitoringStatsReader
from oce.infrastructure.persistence.models import (
    ApiCallMetricModel,
    ResourceSampleModel,
    RetrievalMetricModel,
    TokenUsageMetricModel,
)
from oce.shared.database.session import Base


async def _make_factory():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    return async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False), engine


def _recent() -> datetime:
    return datetime.now(timezone.utc) - timedelta(minutes=10)


def _old() -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=2)


async def _seed(factory) -> None:
    async with factory() as session:
        session.add_all([
            ApiCallMetricModel(ts=_recent(), endpoint="/a", method="GET", status_code=200, latency_ms=10),
            ApiCallMetricModel(ts=_recent(), endpoint="/a", method="GET", status_code=200, latency_ms=20),
            ApiCallMetricModel(ts=_recent(), endpoint="/a", method="GET", status_code=500, latency_ms=30),
            ApiCallMetricModel(ts=_old(), endpoint="/a", method="GET", status_code=200, latency_ms=999),
            TokenUsageMetricModel(ts=_recent(), kind="embed", model="m", prompt_tokens=100, total_tokens=100),
            TokenUsageMetricModel(ts=_recent(), kind="embed", model="m", prompt_tokens=50, total_tokens=50),
            TokenUsageMetricModel(ts=_recent(), kind="rerank", model="r", total_tokens=20),
            TokenUsageMetricModel(ts=_old(), kind="embed", model="m", total_tokens=777),
            RetrievalMetricModel(ts=_recent(), source="retrieval", hit_count=3, total_ms=5),
            RetrievalMetricModel(ts=_recent(), source="retrieval", hit_count=0, total_ms=4),
            RetrievalMetricModel(ts=_recent(), source="overview", hit_count=2, total_ms=6),
            RetrievalMetricModel(ts=_old(), source="retrieval", hit_count=0, total_ms=1),
            ResourceSampleModel(
                ts=_recent(), disk_data_bytes=11, disk_free_bytes=22, disk_total_bytes=33,
                mem_rss_bytes=44, mem_percent=5.5, cpu_percent=6.6,
            ),
        ])
        await session.commit()


async def test_reader_aggregates_within_window():
    factory, engine = await _make_factory()
    try:
        await _seed(factory)
        stats = await SqlMonitoringStatsReader(factory).read(window_hours=1)

        # api：窗口外 999 被排除；[10,20,30] → avg20/p50=20/p95=30/max30，1 个 5xx
        assert stats.api_calls.count == 3
        assert stats.api_calls.error_count == 1
        assert stats.api_calls.avg_latency_ms == 20.0
        assert stats.api_calls.p50_latency_ms == 20
        assert stats.api_calls.p95_latency_ms == 30
        assert stats.api_calls.max_latency_ms == 30

        # token：按 kind 聚合，窗口外 777 被排除
        by_kind = {t.kind: t for t in stats.tokens}
        assert by_kind["embed"].calls == 2
        assert by_kind["embed"].prompt_tokens == 150
        assert by_kind["embed"].total_tokens == 150
        assert by_kind["rerank"].total_tokens == 20
        assert stats.tokens_total == 170

        # retrieval：窗口内 3 条，1 条空回
        assert stats.retrieval.count == 3
        assert stats.retrieval.empty_count == 1
        assert stats.retrieval.empty_rate == round(1 / 3, 4)

        # resource：返回最新快照
        assert stats.resource is not None
        assert stats.resource.disk_total_bytes == 33
        assert stats.resource.cpu_percent == 6.6
    finally:
        await engine.dispose()


async def test_reader_empty_when_no_data():
    factory, engine = await _make_factory()
    try:
        stats = await SqlMonitoringStatsReader(factory).read(window_hours=24)
        assert stats.api_calls.count == 0
        assert stats.api_calls.p95_latency_ms == 0
        assert stats.tokens == ()
        assert stats.tokens_total == 0
        assert stats.retrieval.count == 0
        assert stats.retrieval.empty_rate == 0.0
        assert stats.resource is None
    finally:
        await engine.dispose()
