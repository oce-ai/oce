"""SqlReportsReader 单测：8 个报表聚合的窗口过滤/分桶/分组/降级行为。"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

import oce.infrastructure.persistence.models  # noqa: F401  注册 ORM 表到 Base.metadata
from oce.infrastructure.metrics.report_store import SqlReportsReader
from oce.infrastructure.persistence.models import (
    ApiCallMetricModel,
    BlobChunkModel,
    BlobModel,
    BlobStagingModel,
    ChainModel,
    ChunkModel,
    ResourceSampleModel,
    RetrievalMetricModel,
    SymbolOccurrenceModel,
    TokenUsageMetricModel,
)
from oce.shared.database.session import Base
from oce.shared.reports_read import VectorCollectionStat, VectorStoreStat


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


async def _add(factory, rows) -> None:
    async with factory() as session:
        session.add_all(rows)
        await session.commit()


async def test_api_calls_buckets_endpoints_errors_window():
    factory, engine = await _make_factory()
    try:
        await _add(factory, [
            ApiCallMetricModel(ts=_recent(), endpoint="/a", method="GET", status_code=200, latency_ms=10),
            ApiCallMetricModel(ts=_recent(), endpoint="/a", method="GET", status_code=200, latency_ms=20),
            ApiCallMetricModel(ts=_recent(), endpoint="/a", method="GET", status_code=500, latency_ms=30, error_type="boom"),
            ApiCallMetricModel(ts=_recent(), endpoint="/b", method="POST", status_code=404, latency_ms=40),
            ApiCallMetricModel(ts=_old(), endpoint="/a", method="GET", status_code=200, latency_ms=999),
        ])
        report = await SqlReportsReader(factory).api_calls(window_hours=1, bucket="hour")

        # 分桶：窗口外 999 被排除；4 条同一小时桶
        assert report.window_hours == 1 and report.bucket == "hour"
        assert sum(b.count for b in report.buckets) == 4
        assert sum(b.error_count for b in report.buckets) == 1  # 仅 5xx
        assert max(b.max_latency_ms for b in report.buckets) == 40

        # 端点分组：按 count 降序
        by_ep = {(e.endpoint, e.method): e for e in report.endpoints}
        assert by_ep[("/a", "GET")].count == 3
        assert by_ep[("/a", "GET")].error_count == 1
        assert by_ep[("/a", "GET")].error_rate == round(1 / 3, 4)
        assert by_ep[("/b", "POST")].count == 1
        assert by_ep[("/b", "POST")].error_count == 0  # 404 不计入 5xx
        assert report.endpoints[0].endpoint == "/a"

        # 错误明细：status>=400 都进 errors
        by_err = {(e.status_code, e.error_type): e for e in report.errors}
        assert by_err[(500, "boom")].count == 1
        assert by_err[(404, None)].count == 1
        assert by_err[(500, "boom")].last_ts is not None
    finally:
        await engine.dispose()


async def test_retrieval_buckets_stages_intents_scopes():
    factory, engine = await _make_factory()
    try:
        await _add(factory, [
            RetrievalMetricModel(ts=_recent(), source="retrieval", hit_count=3, total_ms=10,
                                 intent="code", scope_size=50, dense_ms=5, path_boosted=True),
            RetrievalMetricModel(ts=_recent(), source="retrieval", hit_count=0, total_ms=20,
                                 intent="code", scope_size=500, dense_ms=7, path_boosted=False),
            RetrievalMetricModel(ts=_recent(), source="overview", hit_count=2, total_ms=30,
                                 path_boosted=False),
            RetrievalMetricModel(ts=_old(), source="retrieval", hit_count=0, total_ms=999,
                                 intent="code", path_boosted=False),
        ])
        report = await SqlReportsReader(factory).retrieval(window_hours=1, bucket="hour")

        # 分桶：窗口内 3 条，1 条空回
        assert sum(b.count for b in report.buckets) == 3
        assert sum(b.empty_count for b in report.buckets) == 1
        assert max(b.p95_total_ms for b in report.buckets) == 30

        # 阶段：只有有样本的 dense 出现
        assert [s.stage for s in report.stages] == ["dense"]
        assert report.stages[0].count == 2
        assert report.stages[0].avg_ms == 6.0
        assert report.stages[0].max_ms == 7

        # intent 分组（含 None），path_boosted 计数
        by_intent = {i.intent: i for i in report.intents}
        assert by_intent["code"].count == 2
        assert by_intent["code"].empty_count == 1
        assert by_intent["code"].path_boosted_count == 1
        assert by_intent[None].count == 1

        # scope 分桶标签：50→1-100、500→101-1000、None→unknown
        assert [s.label for s in report.scopes] == ["1-100", "101-1000", "unknown"]
        assert all(s.count == 1 for s in report.scopes)
    finally:
        await engine.dispose()


async def test_slow_queries_ordered_by_total_ms_desc():
    factory, engine = await _make_factory()
    try:
        await _add(factory, [
            RetrievalMetricModel(ts=_recent(), source="retrieval", hit_count=1, total_ms=100, query_text="q1"),
            RetrievalMetricModel(ts=_recent(), source="retrieval", hit_count=1, total_ms=300, query_text="q3"),
            RetrievalMetricModel(ts=_recent(), source="retrieval", hit_count=1, total_ms=200, query_text="q2"),
            RetrievalMetricModel(ts=_old(), source="retrieval", hit_count=1, total_ms=999, query_text="old"),
        ])
        reader = SqlReportsReader(factory)
        items = await reader.slow_queries(window_hours=1, limit=10)
        assert [i.total_ms for i in items] == [300, 200, 100]  # 窗口外 999 排除
        assert items[0].query_text == "q3"

        limited = await reader.slow_queries(window_hours=1, limit=2)
        assert [i.total_ms for i in limited] == [300, 200]
    finally:
        await engine.dispose()


async def test_empty_queries_filters_hits_and_orders_ts_desc():
    factory, engine = await _make_factory()
    try:
        earlier = datetime.now(timezone.utc) - timedelta(minutes=30)
        await _add(factory, [
            RetrievalMetricModel(ts=earlier, source="retrieval", hit_count=0, total_ms=5, query_text="e1"),
            RetrievalMetricModel(ts=_recent(), source="retrieval", hit_count=0, total_ms=6, query_text="e2"),
            RetrievalMetricModel(ts=_recent(), source="retrieval", hit_count=3, total_ms=7, query_text="hit"),
            RetrievalMetricModel(ts=_old(), source="retrieval", hit_count=0, total_ms=8, query_text="old"),
        ])
        items = await SqlReportsReader(factory).empty_queries(window_hours=1, limit=10)
        assert [i.query_text for i in items] == ["e2", "e1"]  # 仅空回、ts 降序、窗口外排除
        assert all(i.hit_count == 0 for i in items)
    finally:
        await engine.dispose()


async def test_tokens_buckets_models_credentials_total():
    factory, engine = await _make_factory()
    try:
        await _add(factory, [
            TokenUsageMetricModel(ts=_recent(), kind="embed", model="m1", credential_id=1,
                                  prompt_tokens=100, completion_tokens=0, total_tokens=100),
            TokenUsageMetricModel(ts=_recent(), kind="embed", model="m1", credential_id=1,
                                  prompt_tokens=50, completion_tokens=0, total_tokens=50),
            TokenUsageMetricModel(ts=_recent(), kind="rerank", model="r1", credential_id=None,
                                  prompt_tokens=10, completion_tokens=10, total_tokens=20),
            TokenUsageMetricModel(ts=_old(), kind="embed", model="m1", total_tokens=777),
        ])
        report = await SqlReportsReader(factory).tokens(window_hours=1, bucket="hour")

        # 分桶按 (ts, kind)：窗口内两个 kind 各一桶
        by_kind = {b.kind: b for b in report.buckets}
        assert by_kind["embed"].calls == 2
        assert by_kind["embed"].prompt_tokens == 150
        assert by_kind["embed"].total_tokens == 150
        assert by_kind["rerank"].completion_tokens == 10

        # 模型排行按 total_tokens 降序
        assert [(m.model, m.kind) for m in report.models] == [("m1", "embed"), ("r1", "rerank")]
        assert report.models[0].avg_tokens_per_call == 75.0

        # 凭据分组含 credential_id=None（环境变量回退）
        by_cred = {c.credential_id: c for c in report.credentials}
        assert by_cred[1].calls == 2 and by_cred[1].total_tokens == 150
        assert by_cred[None].total_tokens == 20

        assert report.tokens_total == 170  # 窗口外 777 排除
    finally:
        await engine.dispose()


async def test_index_inventory_counts():
    factory, engine = await _make_factory()
    try:
        await _add(factory, [
            BlobModel(blob_name="b1", path="a.py", content_size=100, language="python",
                      status="indexed"),
            BlobModel(blob_name="b2", path="b.md", content_size=50, status="pending",
                      retry_count=2),
            BlobStagingModel(blob_name="b2", content="raw"),
            ChunkModel(content_hash="c1", content="x", content_size=10,
                       chunk_type="code", embedded=True),
            ChunkModel(content_hash="c2", content="y", content_size=20, embedded=False),
            ChainModel(chain_id="ch1", updated_at=datetime.now(timezone.utc) - timedelta(days=10)),
        ])
        await _add(factory, [
            BlobChunkModel(blob_name="b1", content_hash="c1", start_line=1, end_line=2, chunk_index=0),
            SymbolOccurrenceModel(identifier="foo", blob_name="b1", content_hash="c1",
                                  kind="def", start_line=1, end_line=2),
        ])
        report = await SqlReportsReader(factory).index_inventory()

        assert report.blob_total == 2
        assert report.blob_content_bytes == 150
        assert report.blob_retrying == 1
        assert {c.key: c.count for c in report.blob_by_status} == {"indexed": 1, "pending": 1}
        assert {c.key: c.count for c in report.blob_by_language} == {"python": 1, "unknown": 1}
        assert report.chunk_total == 2
        assert report.chunk_pending_embed == 1
        assert report.chunk_content_bytes == 30
        assert {c.key: c.count for c in report.chunk_by_type} == {"code": 1, "unknown": 1}
        assert report.blob_chunk_links == 1
        assert report.symbol_total == 1
        assert {c.key: c.count for c in report.symbol_by_kind} == {"def": 1}
        assert report.chain_total == 1
        assert report.chain_stale_7d == 1
        assert report.chain_stale_30d == 0
        assert report.staging_rows == 1
    finally:
        await engine.dispose()


async def test_resources_buckets_and_disk_growth():
    factory, engine = await _make_factory()
    try:
        now = datetime.now(timezone.utc)
        # 两个样本相隔 25h（0.5GB 增长）→ 日增速 = 增量 / (25/24 天)
        await _add(factory, [
            ResourceSampleModel(ts=now - timedelta(hours=25), cpu_percent=10.0, mem_percent=20.0,
                                mem_rss_bytes=100, disk_data_bytes=1000,
                                disk_free_bytes=9000, disk_total_bytes=10000),
            ResourceSampleModel(ts=now - timedelta(minutes=10), cpu_percent=30.0, mem_percent=40.0,
                                mem_rss_bytes=200, disk_data_bytes=3400,
                                disk_free_bytes=6600, disk_total_bytes=10000),
        ])
        report = await SqlReportsReader(factory).resources(window_hours=48, bucket="hour")

        assert len(report.buckets) == 2  # 两个不同的小时桶
        last = report.buckets[-1]
        assert last.avg_cpu_percent == 30.0 and last.max_cpu_percent == 30.0
        assert last.max_mem_rss_bytes == 200
        assert last.disk_data_bytes == 3400 and last.disk_free_bytes == 6600
        assert report.disk_total_bytes == 10000

        elapsed_days = (25 * 3600 - 10 * 60) / 86400
        assert report.disk_growth_bytes_per_day == round(2400 / elapsed_days, 2)
        assert report.disk_days_until_full == round(6600 / report.disk_growth_bytes_per_day, 1)
    finally:
        await engine.dispose()


async def test_resources_growth_zero_with_single_sample():
    factory, engine = await _make_factory()
    try:
        await _add(factory, [
            ResourceSampleModel(ts=_recent(), cpu_percent=5.0, mem_percent=6.0,
                                mem_rss_bytes=1, disk_data_bytes=2,
                                disk_free_bytes=3, disk_total_bytes=4),
        ])
        report = await SqlReportsReader(factory).resources(window_hours=1, bucket="hour")
        assert report.disk_growth_bytes_per_day == 0.0
        assert report.disk_days_until_full is None
        assert report.disk_total_bytes == 4
    finally:
        await engine.dispose()


async def test_storage_dialect_tables_and_data_dir(tmp_path):
    factory, engine = await _make_factory()
    try:
        await _add(factory, [
            BlobModel(blob_name="b1", path="a.py", content_size=1),
        ])
        (tmp_path / "oce.db").write_bytes(b"x" * 128)
        report = await SqlReportsReader(factory, data_dir=str(tmp_path)).storage()

        assert report.dialect == "sqlite"
        assert len(report.tables) == 12
        by_table = {t.table: t for t in report.tables}
        assert by_table["blobs"].rows == 1
        assert by_table["chunks"].rows == 0

        assert report.data_dir == str(tmp_path)
        assert [f.name for f in report.data_files] == ["oce.db"]
        assert report.data_files[0].bytes == 128
        assert report.data_dir_total_bytes == 128
    finally:
        await engine.dispose()


async def test_storage_vector_none_without_provider():
    factory, engine = await _make_factory()
    try:
        report = await SqlReportsReader(factory).storage()
        assert report.vector is None
    finally:
        await engine.dispose()


async def test_storage_vector_included_from_provider():
    factory, engine = await _make_factory()
    try:
        stat = VectorStoreStat(
            mode="lite",
            collections=(
                VectorCollectionStat(name="oce_chunks", rows=10, est_bytes=40960),
            ),
            file_bytes=2048,
        )

        async def _provider() -> VectorStoreStat:
            return stat

        report = await SqlReportsReader(factory, vector_stats=_provider).storage()
        assert report.vector == stat
        assert report.vector.collections[0].rows == 10
    finally:
        await engine.dispose()


async def test_storage_vector_degrades_to_unavailable_on_error():
    factory, engine = await _make_factory()
    try:
        async def _provider() -> VectorStoreStat:
            raise RuntimeError("milvus down")

        report = await SqlReportsReader(factory, vector_stats=_provider).storage()
        assert report.vector is not None
        assert report.vector.mode == "unavailable"
        assert report.vector.error == "milvus down"
        assert report.vector.collections == ()
    finally:
        await engine.dispose()


async def test_all_reports_empty_when_no_data(tmp_path):
    factory, engine = await _make_factory()
    try:
        reader = SqlReportsReader(factory)
        api = await reader.api_calls(window_hours=24, bucket="day")
        assert api.buckets == () and api.endpoints == () and api.errors == ()

        retrieval = await reader.retrieval(window_hours=24, bucket="day")
        assert retrieval.buckets == () and retrieval.stages == ()
        assert retrieval.intents == () and retrieval.scopes == ()

        assert await reader.slow_queries(window_hours=24, limit=10) == ()
        assert await reader.empty_queries(window_hours=24, limit=10) == ()

        tokens = await reader.tokens(window_hours=24, bucket="day")
        assert tokens.buckets == () and tokens.tokens_total == 0

        inventory = await reader.index_inventory()
        assert inventory.blob_total == 0 and inventory.chunk_total == 0
        assert inventory.staging_rows == 0

        resources = await reader.resources(window_hours=24, bucket="day")
        assert resources.buckets == () and resources.disk_total_bytes == 0
        assert resources.disk_days_until_full is None

        storage = await reader.storage()
        assert storage.dialect == "sqlite"
        assert all(t.rows == 0 for t in storage.tables)
        assert storage.data_dir is None and storage.data_files == ()
    finally:
        await engine.dispose()
