"""MonitoringCleaner 单测：按 retention_days 删过期监控行，保留期内保留，旁路容错。

用 StaticPool 内存库让多个 session 共享一条连接（默认 :memory: 每连接独立库）。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import oce.infrastructure.persistence.models  # noqa: F401  注册 ORM 表到 Base.metadata
from oce.infrastructure.metrics.cleanup import MonitoringCleaner
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
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    return factory, engine


def _old() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=40)


def _recent() -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=1)


async def _count(factory, model) -> int:
    async with factory() as session:
        return await session.scalar(select(func.count()).select_from(model))


async def test_cleanup_deletes_expired_keeps_recent():
    factory, engine = await _make_factory()
    try:
        async with factory() as session:
            session.add_all([
                ApiCallMetricModel(ts=_old(), endpoint="/x", method="GET", status_code=200, latency_ms=1),
                ApiCallMetricModel(ts=_recent(), endpoint="/y", method="GET", status_code=200, latency_ms=1),
                TokenUsageMetricModel(ts=_old(), kind="embed", model="m", total_tokens=1),
                ResourceSampleModel(
                    ts=_old(), disk_data_bytes=1, disk_free_bytes=2, disk_total_bytes=3,
                    mem_rss_bytes=4, mem_percent=5.0, cpu_percent=6.0,
                ),
                RetrievalMetricModel(ts=_recent(), source="retrieval", hit_count=1, total_ms=5),
            ])
            await session.commit()

        cleaner = MonitoringCleaner(factory, retention_days=30, interval_seconds=999)
        removed = await cleaner._cleanup_once()

        assert removed == 3  # 三条 40 天前的（api/token/resource）
        assert await _count(factory, ApiCallMetricModel) == 1   # recent 保留
        assert await _count(factory, TokenUsageMetricModel) == 0
        assert await _count(factory, ResourceSampleModel) == 0
        assert await _count(factory, RetrievalMetricModel) == 1  # recent 保留
    finally:
        await engine.dispose()


async def test_cleanup_swallows_errors():
    """session_factory 抛错 → 清理返回 0，不上抛（旁路容错）。"""
    def _boom():
        raise RuntimeError("db down")

    cleaner = MonitoringCleaner(_boom, retention_days=30, interval_seconds=999)
    assert await cleaner._cleanup_once() == 0


async def test_loop_runs_cleanup_periodically():
    factory, engine = await _make_factory()
    try:
        async with factory() as session:
            session.add(
                ApiCallMetricModel(ts=_old(), endpoint="/x", method="GET", status_code=200, latency_ms=1)
            )
            await session.commit()

        cleaner = MonitoringCleaner(factory, retention_days=30, interval_seconds=0.01)
        await cleaner.start()
        await asyncio.sleep(0.05)
        await cleaner.stop()

        assert await _count(factory, ApiCallMetricModel) == 0  # 循环跑过并清掉过期行
    finally:
        await engine.dispose()
