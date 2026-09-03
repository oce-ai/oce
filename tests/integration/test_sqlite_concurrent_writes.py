"""集成测试：验证 SQLite WAL 模式下监控指标并发写入不会死锁。

模拟个人模式场景：后台 metrics sink flush + 多个并发 session 写入。
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from oce.infrastructure.metrics.sql_metrics_sink import SqlMetricsSink
from oce.infrastructure.persistence.models import (
    ApiCallMetricModel,
    ResourceSampleModel,
)
from oce.shared.database.session import Base
from oce.shared.database.sqlite_adapter import SQLiteAdapter
from oce.shared.metrics import ApiCallRecord, ResourceSampleRecord


async def test_concurrent_writes_no_lock_conflict():
    """并发写入 API 调用记录和资源采样，验证不会出现 database locked 错误。"""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        url = f"sqlite+aiosqlite:///{db_path.as_posix()}"

        # 使用修复后的 SQLite 适配器
        engine = SQLiteAdapter().create_engine(url, echo=False)
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

        # 创建 metrics sink（短 flush 间隔模拟高频写入）
        sink = SqlMetricsSink(factory, flush_interval_seconds=0.1, max_buffer=100)
        await sink.start()

        try:
            # 并发写入 API 调用记录和资源采样
            for i in range(50):
                # API 调用记录
                sink.record_api_call(ApiCallRecord(
                    ts=datetime.now(timezone.utc),
                    endpoint=f"/test/{i}",
                    method="GET",
                    status_code=200,
                    latency_ms=10 + i,
                    error_type=None,
                ))
                # 资源采样
                if i % 5 == 0:
                    sink.record_resource_sample(ResourceSampleRecord(
                        disk_data_bytes=1024 * i,
                        disk_free_bytes=1024 * 1024 * 100,
                        disk_total_bytes=1024 * 1024 * 500,
                        mem_rss_bytes=1024 * 1024 * 50,
                        mem_percent=10.0,
                        cpu_percent=5.0,
                    ))
                # 每 10 条休眠一下，让 flush 有机会并发执行
                if i % 10 == 0:
                    await asyncio.sleep(0.05)

            # 等待所有写入完成
            await asyncio.sleep(1.0)

        finally:
            await sink.stop()
            await engine.dispose()

        # 验证数据写入成功（重新连接以确保 WAL 已关闭）
        verify_engine = SQLiteAdapter().create_engine(url, echo=False)
        verify_factory = async_sessionmaker(verify_engine, class_=AsyncSession, expire_on_commit=False)

        async with verify_factory() as session:
            api_count = await session.scalar(
                select(func.count()).select_from(ApiCallMetricModel)
            )
            resource_count = await session.scalar(
                select(func.count()).select_from(ResourceSampleModel)
            )

        await verify_engine.dispose()

        # 应该至少写入了大部分数据（允许 deque maxlen 丢弃少量）
        assert api_count >= 45, f"Expected >=45 API records, got {api_count}"
        assert resource_count >= 8, f"Expected >=8 resource samples, got {resource_count}"
        print(f"✓ Wrote {api_count} API calls and {resource_count} resource samples without lock errors")
