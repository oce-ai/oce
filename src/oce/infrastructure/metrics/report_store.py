"""报表只读聚合的 SQL 实现（跨 SQLite / PostgreSQL 可移植）。

报表是旁路只读路径，绝不写库、不影响主链路。为保证跨方言可移植：
- 不使用方言专有的日期截断 / percentile 函数；SQL 只做时间窗口过滤与基础聚合，
  分桶与分位数在 Python 侧计算（窗口内行数有限，内存代价可接受）；
- 表空间占用按方言分支：PG 用 pg_total_relation_size，SQLite 尝试 dbstat 虚表，
  不可用时回退为 0 并标记 approximate=True；
- 个人模式下 data_dir 目录体积用同步文件系统 IO 统计（管理员旁路、目录很小，可接受）。
"""
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from oce.infrastructure.persistence.models import (
    ApiCallMetricModel,
    BlobChunkModel,
    BlobModel,
    BlobStagingModel,
    ChainMemberModel,
    ChainModel,
    ChunkModel,
    ModelCredentialModel,
    ResourceSampleModel,
    RetrievalMetricModel,
    SymbolOccurrenceModel,
    TokenUsageMetricModel,
)
from oce.shared.reports_read import (
    ApiCallBucket,
    ApiCallsReport,
    CountStat,
    CredentialTokenStat,
    DataFileStat,
    EndpointStat,
    ErrorStat,
    IndexInventoryReport,
    IntentStat,
    ModelTokenStat,
    ResourceBucket,
    ResourcesReport,
    RetrievalBucket,
    RetrievalQueryDetail,
    RetrievalReport,
    ScopeBucketStat,
    StageStat,
    StorageReport,
    TableSpaceStat,
    TokenBucket,
    TokensReport,
    VectorStoreStat,
)

# 检索管线阶段名（对应 retrieval_metrics 的 <stage>_ms 列）。
_STAGE_NAMES: tuple[str, ...] = (
    "intent",
    "rewrite",
    "dense",
    "exact",
    "fuse",
    "rerank",
    "llm_rerank",
    "select",
)

# 工作集规模分桶标签（逻辑顺序即输出顺序）。
_SCOPE_LABELS: tuple[str, ...] = ("1-100", "101-1000", "1001-10000", ">10000", "unknown")

# storage() 关心的全部业务表：(表名, ORM 模型)，用于逐表统计空间与行数。
_SPACE_TABLES: tuple[tuple[str, type], ...] = (
    ("model_credentials", ModelCredentialModel),
    ("blobs", BlobModel),
    ("blob_staging", BlobStagingModel),
    ("chunks", ChunkModel),
    ("blob_chunks", BlobChunkModel),
    ("chains", ChainModel),
    ("chain_members", ChainMemberModel),
    ("symbol_occurrences", SymbolOccurrenceModel),
    ("api_call_metrics", ApiCallMetricModel),
    ("token_usage_metrics", TokenUsageMetricModel),
    ("resource_samples", ResourceSampleModel),
    ("retrieval_metrics", RetrievalMetricModel),
)


def _percentile(sorted_vals: list[int], p: int) -> int:
    if not sorted_vals:
        return 0
    idx = int(round((p / 100) * (len(sorted_vals) - 1)))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


def _check_bucket(bucket: str) -> str:
    """校验分桶粒度；调用方（API 层）已做过 sanitize，这里兜底防御。"""
    if bucket not in ("hour", "day"):
        raise ValueError(f"bucket 必须是 'hour' 或 'day'，收到: {bucket!r}")
    return bucket


def _bucket_ts(ts: datetime, bucket: str) -> datetime:
    """把时间戳截断到分桶边界；naive 时间视为 UTC。"""
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    ts = ts.replace(minute=0, second=0, microsecond=0)
    if bucket == "day":
        ts = ts.replace(hour=0)
    return ts


def _cutoff(window_hours: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=window_hours)


def _scope_label(scope_size: int | None) -> str:
    if scope_size is None:
        return "unknown"
    if scope_size <= 100:
        return "1-100"
    if scope_size <= 1000:
        return "101-1000"
    if scope_size <= 10000:
        return "1001-10000"
    return ">10000"


def _query_detail(row: RetrievalMetricModel) -> RetrievalQueryDetail:
    return RetrievalQueryDetail(
        ts=row.ts,
        source=row.source,
        query_text=row.query_text,
        total_ms=int(row.total_ms),
        hit_count=int(row.hit_count),
        scope_size=row.scope_size,
        intent=row.intent,
        path_boosted=bool(row.path_boosted),
    )


def _entry_size_bytes(entry: Path) -> int:
    """单文件取 st_size；目录递归累加（跳过读不到的文件）。"""
    try:
        if entry.is_file():
            return entry.stat().st_size
        total = 0
        for root, _dirs, files in os.walk(entry):
            for name in files:
                try:
                    total += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
        return total
    except OSError:
        return 0


class SqlReportsReader:
    """报表聚合读端口的 SQL 实现；SQL 只做窗口过滤，分桶/分位在 Python 侧算。"""

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        *,
        data_dir: str | None = None,
        vector_stats: Callable[[], Awaitable[VectorStoreStat]] | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._data_dir = data_dir
        self._vector_stats = vector_stats

    # ------------------------------------------------------------ API 健康

    async def api_calls(self, window_hours: int, bucket: str) -> ApiCallsReport:
        _check_bucket(bucket)
        cutoff = _cutoff(window_hours)
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(
                        ApiCallMetricModel.ts,
                        ApiCallMetricModel.endpoint,
                        ApiCallMetricModel.method,
                        ApiCallMetricModel.status_code,
                        ApiCallMetricModel.latency_ms,
                        ApiCallMetricModel.error_type,
                    ).where(ApiCallMetricModel.ts >= cutoff)
                )
            ).all()

        by_bucket: dict[datetime, list[tuple[int, int]]] = defaultdict(list)
        by_endpoint: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
        by_error: dict[tuple[int, str | None], list[datetime]] = defaultdict(list)
        for ts, endpoint, method, status_code, latency_ms, error_type in rows:
            status = int(status_code)
            latency = int(latency_ms)
            by_bucket[_bucket_ts(ts, bucket)].append((latency, status))
            by_endpoint[(endpoint, method)].append((latency, status))
            if status >= 400:
                by_error[(status, error_type)].append(ts)

        buckets = []
        for bts in sorted(by_bucket):
            samples = by_bucket[bts]
            latencies = sorted(latency for latency, _status in samples)
            count = len(latencies)
            buckets.append(
                ApiCallBucket(
                    ts=bts,
                    count=count,
                    error_count=sum(1 for _l, status in samples if status >= 500),
                    avg_latency_ms=round(sum(latencies) / count, 2),
                    p50_latency_ms=_percentile(latencies, 50),
                    p95_latency_ms=_percentile(latencies, 95),
                    max_latency_ms=latencies[-1],
                )
            )

        endpoints = []
        for (endpoint, method), samples in by_endpoint.items():
            latencies = sorted(latency for latency, _status in samples)
            count = len(latencies)
            error_count = sum(1 for _l, status in samples if status >= 500)
            endpoints.append(
                EndpointStat(
                    endpoint=endpoint,
                    method=method,
                    count=count,
                    error_count=error_count,
                    error_rate=round(error_count / count, 4),
                    avg_latency_ms=round(sum(latencies) / count, 2),
                    p95_latency_ms=_percentile(latencies, 95),
                )
            )
        endpoints.sort(key=lambda e: e.count, reverse=True)

        errors = [
            ErrorStat(
                status_code=status,
                error_type=error_type,
                count=len(ts_list),
                last_ts=max(ts_list),
            )
            for (status, error_type), ts_list in by_error.items()
        ]
        errors.sort(key=lambda e: e.count, reverse=True)

        return ApiCallsReport(
            window_hours=window_hours,
            bucket=bucket,
            buckets=tuple(buckets),
            endpoints=tuple(endpoints[:50]),
            errors=tuple(errors[:50]),
        )

    # ------------------------------------------------------------ 检索质量

    async def retrieval(self, window_hours: int, bucket: str) -> RetrievalReport:
        _check_bucket(bucket)
        cutoff = _cutoff(window_hours)
        stage_cols = [
            getattr(RetrievalMetricModel, f"{stage}_ms") for stage in _STAGE_NAMES
        ]
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(
                        RetrievalMetricModel.ts,
                        RetrievalMetricModel.scope_size,
                        RetrievalMetricModel.hit_count,
                        RetrievalMetricModel.total_ms,
                        RetrievalMetricModel.intent,
                        RetrievalMetricModel.path_boosted,
                        *stage_cols,
                    ).where(RetrievalMetricModel.ts >= cutoff)
                )
            ).all()

        # (hit_count, total_ms) 元组即可覆盖分桶所需；其余维度单独累计。
        by_bucket: dict[datetime, list[tuple[int, int]]] = defaultdict(list)
        stage_samples: dict[str, list[int]] = defaultdict(list)
        by_intent: dict[str | None, list[tuple[int, int, bool]]] = defaultdict(list)
        by_scope: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for row in rows:
            ts, scope_size, hit_count, total_ms, intent, path_boosted = row[:6]
            hits = int(hit_count)
            total = int(total_ms)
            by_bucket[_bucket_ts(ts, bucket)].append((hits, total))
            by_intent[intent].append((hits, total, bool(path_boosted)))
            by_scope[_scope_label(scope_size)].append((hits, total))
            for stage, value in zip(_STAGE_NAMES, row[6:]):
                if value is not None:
                    stage_samples[stage].append(int(value))

        buckets = []
        for bts in sorted(by_bucket):
            samples = by_bucket[bts]
            totals = sorted(total for _hits, total in samples)
            count = len(samples)
            empty_count = sum(1 for hits, _t in samples if hits == 0)
            buckets.append(
                RetrievalBucket(
                    ts=bts,
                    count=count,
                    empty_count=empty_count,
                    empty_rate=round(empty_count / count, 4),
                    avg_hit_count=round(sum(hits for hits, _t in samples) / count, 2),
                    avg_total_ms=round(sum(totals) / count, 2),
                    p95_total_ms=_percentile(totals, 95),
                )
            )

        stages = []
        for stage in _STAGE_NAMES:
            samples_ms = sorted(stage_samples.get(stage, []))
            if not samples_ms:
                continue  # 零样本阶段直接省略
            stages.append(
                StageStat(
                    stage=stage,
                    count=len(samples_ms),
                    avg_ms=round(sum(samples_ms) / len(samples_ms), 2),
                    p95_ms=_percentile(samples_ms, 95),
                    max_ms=samples_ms[-1],
                )
            )

        intents = []
        for intent, samples3 in by_intent.items():
            count = len(samples3)
            empty_count = sum(1 for hits, _t, _b in samples3 if hits == 0)
            intents.append(
                IntentStat(
                    intent=intent,
                    count=count,
                    empty_count=empty_count,
                    empty_rate=round(empty_count / count, 4),
                    avg_total_ms=round(
                        sum(total for _h, total, _b in samples3) / count, 2
                    ),
                    path_boosted_count=sum(1 for _h, _t, b in samples3 if b),
                )
            )
        intents.sort(key=lambda i: i.count, reverse=True)

        scopes = []
        for label in _SCOPE_LABELS:
            samples = by_scope.get(label)
            if not samples:
                continue  # 空桶省略
            totals = sorted(total for _hits, total in samples)
            count = len(samples)
            scopes.append(
                ScopeBucketStat(
                    label=label,
                    count=count,
                    empty_rate=round(
                        sum(1 for hits, _t in samples if hits == 0) / count, 4
                    ),
                    p95_total_ms=_percentile(totals, 95),
                )
            )

        return RetrievalReport(
            window_hours=window_hours,
            bucket=bucket,
            buckets=tuple(buckets),
            stages=tuple(stages),
            intents=tuple(intents),
            scopes=tuple(scopes),
        )

    async def slow_queries(
        self, window_hours: int, limit: int
    ) -> tuple[RetrievalQueryDetail, ...]:
        cutoff = _cutoff(window_hours)
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(RetrievalMetricModel)
                    .where(RetrievalMetricModel.ts >= cutoff)
                    .order_by(RetrievalMetricModel.total_ms.desc())
                    .limit(limit)
                )
            ).scalars().all()
        return tuple(_query_detail(row) for row in rows)

    async def empty_queries(
        self, window_hours: int, limit: int
    ) -> tuple[RetrievalQueryDetail, ...]:
        cutoff = _cutoff(window_hours)
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(RetrievalMetricModel)
                    .where(
                        RetrievalMetricModel.ts >= cutoff,
                        RetrievalMetricModel.hit_count == 0,
                    )
                    .order_by(RetrievalMetricModel.ts.desc())
                    .limit(limit)
                )
            ).scalars().all()
        return tuple(_query_detail(row) for row in rows)

    # ------------------------------------------------------------ Token 用量

    async def tokens(self, window_hours: int, bucket: str) -> TokensReport:
        _check_bucket(bucket)
        cutoff = _cutoff(window_hours)
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(
                        TokenUsageMetricModel.ts,
                        TokenUsageMetricModel.kind,
                        TokenUsageMetricModel.model,
                        TokenUsageMetricModel.credential_id,
                        TokenUsageMetricModel.prompt_tokens,
                        TokenUsageMetricModel.completion_tokens,
                        TokenUsageMetricModel.total_tokens,
                    ).where(TokenUsageMetricModel.ts >= cutoff)
                )
            ).all()

        # 值元组: (prompt, completion, total)
        by_bucket: dict[tuple[datetime, str], list[tuple[int, int, int]]] = defaultdict(list)
        by_model: dict[tuple[str, str], list[tuple[int, int, int]]] = defaultdict(list)
        by_credential: dict[int | None, list[int]] = defaultdict(list)
        for ts, kind, model, credential_id, prompt, completion, total in rows:
            sums = (int(prompt), int(completion), int(total))
            by_bucket[(_bucket_ts(ts, bucket), kind)].append(sums)
            by_model[(model, kind)].append(sums)
            by_credential[credential_id].append(int(total))

        buckets = tuple(
            TokenBucket(
                ts=bts,
                kind=kind,
                calls=len(samples),
                prompt_tokens=sum(p for p, _c, _t in samples),
                completion_tokens=sum(c for _p, c, _t in samples),
                total_tokens=sum(t for _p, _c, t in samples),
            )
            for (bts, kind), samples in sorted(by_bucket.items())
        )

        models = []
        for (model, kind), samples in by_model.items():
            calls = len(samples)
            total_tokens = sum(t for _p, _c, t in samples)
            models.append(
                ModelTokenStat(
                    model=model,
                    kind=kind,
                    calls=calls,
                    prompt_tokens=sum(p for p, _c, _t in samples),
                    completion_tokens=sum(c for _p, c, _t in samples),
                    total_tokens=total_tokens,
                    avg_tokens_per_call=round(total_tokens / calls, 2),
                )
            )
        models.sort(key=lambda m: m.total_tokens, reverse=True)

        credentials = [
            CredentialTokenStat(
                credential_id=credential_id,
                calls=len(totals),
                total_tokens=sum(totals),
            )
            for credential_id, totals in by_credential.items()
        ]
        credentials.sort(key=lambda c: c.total_tokens, reverse=True)

        return TokensReport(
            window_hours=window_hours,
            bucket=bucket,
            buckets=buckets,
            models=tuple(models),
            credentials=tuple(credentials),
            tokens_total=sum(b.total_tokens for b in buckets),
        )

    # ------------------------------------------------------------ 索引资产

    async def index_inventory(self) -> IndexInventoryReport:
        now = datetime.now(timezone.utc)
        async with self._session_factory() as session:
            blob_row = (
                await session.execute(
                    select(
                        func.count(),
                        func.coalesce(func.sum(BlobModel.content_size), 0),
                    )
                )
            ).one()
            blob_by_status = (
                await session.execute(
                    select(BlobModel.status, func.count()).group_by(BlobModel.status)
                )
            ).all()
            # 注意：不能在 SELECT 和 GROUP BY 各写一次 func.coalesce(...)——两个表达式
            # 对象携带独立绑定参数，asyncpg 编译成 $1/$2 后 PG 判定不等价而报
            # GroupingError。按原始列分组，NULL→"unknown" 的归并放 Python 侧。
            blob_by_language = (
                await session.execute(
                    select(BlobModel.language, func.count()).group_by(
                        BlobModel.language
                    )
                )
            ).all()
            blob_retrying = (
                await session.execute(
                    select(func.count()).where(BlobModel.retry_count > 0)
                )
            ).scalar_one()
            chunk_row = (
                await session.execute(
                    select(
                        func.count(),
                        func.coalesce(func.sum(ChunkModel.content_size), 0),
                    )
                )
            ).one()
            chunk_pending = (
                await session.execute(
                    select(func.count()).where(ChunkModel.embedded.is_(False))
                )
            ).scalar_one()
            chunk_by_type = (
                await session.execute(
                    select(ChunkModel.chunk_type, func.count()).group_by(
                        ChunkModel.chunk_type
                    )
                )
            ).all()
            link_count = (
                await session.execute(
                    select(func.count()).select_from(BlobChunkModel)
                )
            ).scalar_one()
            symbol_total = (
                await session.execute(
                    select(func.count()).select_from(SymbolOccurrenceModel)
                )
            ).scalar_one()
            symbol_by_kind = (
                await session.execute(
                    select(SymbolOccurrenceModel.kind, func.count()).group_by(
                        SymbolOccurrenceModel.kind
                    )
                )
            ).all()
            chain_total = (
                await session.execute(select(func.count()).select_from(ChainModel))
            ).scalar_one()
            chain_stale_7d = (
                await session.execute(
                    select(func.count()).where(
                        ChainModel.updated_at < now - timedelta(days=7)
                    )
                )
            ).scalar_one()
            chain_stale_30d = (
                await session.execute(
                    select(func.count()).where(
                        ChainModel.updated_at < now - timedelta(days=30)
                    )
                )
            ).scalar_one()
            staging_rows = (
                await session.execute(
                    select(func.count()).select_from(BlobStagingModel)
                )
            ).scalar_one()

        languages = sorted(
            (
                CountStat(key=key or "unknown", count=int(count))
                for key, count in blob_by_language
            ),
            key=lambda c: c.count,
            reverse=True,
        )
        return IndexInventoryReport(
            blob_total=int(blob_row[0]),
            blob_by_status=tuple(
                CountStat(key=key, count=int(count)) for key, count in blob_by_status
            ),
            blob_by_language=tuple(languages[:30]),
            blob_retrying=int(blob_retrying),
            blob_content_bytes=int(blob_row[1]),
            chunk_total=int(chunk_row[0]),
            chunk_pending_embed=int(chunk_pending),
            chunk_by_type=tuple(
                CountStat(key=key or "unknown", count=int(count))
                for key, count in chunk_by_type
            ),
            chunk_content_bytes=int(chunk_row[1]),
            blob_chunk_links=int(link_count),
            symbol_total=int(symbol_total),
            symbol_by_kind=tuple(
                CountStat(key=key, count=int(count)) for key, count in symbol_by_kind
            ),
            chain_total=int(chain_total),
            chain_stale_7d=int(chain_stale_7d),
            chain_stale_30d=int(chain_stale_30d),
            staging_rows=int(staging_rows),
        )

    # ------------------------------------------------------------ 资源容量

    async def resources(self, window_hours: int, bucket: str) -> ResourcesReport:
        _check_bucket(bucket)
        cutoff = _cutoff(window_hours)
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(
                        ResourceSampleModel.ts,
                        ResourceSampleModel.cpu_percent,
                        ResourceSampleModel.mem_percent,
                        ResourceSampleModel.mem_rss_bytes,
                        ResourceSampleModel.disk_data_bytes,
                        ResourceSampleModel.disk_free_bytes,
                    )
                    .where(ResourceSampleModel.ts >= cutoff)
                    .order_by(ResourceSampleModel.ts.asc())
                )
            ).all()
            latest = (
                await session.execute(
                    select(
                        ResourceSampleModel.disk_total_bytes,
                        ResourceSampleModel.disk_free_bytes,
                    )
                    .order_by(ResourceSampleModel.ts.desc())
                    .limit(1)
                )
            ).first()

        # rows 已按 ts 升序，桶内最后一行即该桶最新磁盘快照。
        by_bucket: dict[datetime, list[tuple]] = defaultdict(list)
        for row in rows:
            by_bucket[_bucket_ts(row[0], bucket)].append(row)

        buckets = []
        for bts in sorted(by_bucket):
            samples = by_bucket[bts]
            count = len(samples)
            last = samples[-1]
            buckets.append(
                ResourceBucket(
                    ts=bts,
                    avg_cpu_percent=round(sum(r[1] for r in samples) / count, 2),
                    max_cpu_percent=max(r[1] for r in samples),
                    avg_mem_percent=round(sum(r[2] for r in samples) / count, 2),
                    max_mem_rss_bytes=max(int(r[3]) for r in samples),
                    disk_data_bytes=int(last[4]),
                    disk_free_bytes=int(last[5]),
                )
            )

        growth_per_day = 0.0
        if len(rows) >= 2:
            first_row, last_row = rows[0], rows[-1]
            elapsed_seconds = (last_row[0] - first_row[0]).total_seconds()
            if elapsed_seconds >= 3600:
                elapsed_days = elapsed_seconds / 86400
                growth_per_day = round(
                    (int(last_row[4]) - int(first_row[4])) / elapsed_days, 2
                )

        days_until_full: float | None = None
        if growth_per_day > 0 and latest is not None:
            days_until_full = round(int(latest[1]) / growth_per_day, 1)

        return ResourcesReport(
            window_hours=window_hours,
            bucket=bucket,
            buckets=tuple(buckets),
            disk_total_bytes=int(latest[0]) if latest is not None else 0,
            disk_growth_bytes_per_day=growth_per_day,
            disk_days_until_full=days_until_full,
        )

    # ------------------------------------------------------------ 空间占用

    async def storage(self) -> StorageReport:
        tables: list[TableSpaceStat] = []
        dialect = ""
        async with self._session_factory() as session:
            dialect = session.get_bind().dialect.name
            for table_name, model in _SPACE_TABLES:
                row_count = (
                    await session.execute(select(func.count()).select_from(model))
                ).scalar_one()
                size_bytes = 0
                approximate = False
                if dialect == "postgresql":
                    size_bytes = int(
                        (
                            await session.execute(
                                text("SELECT pg_total_relation_size(:tname)"),
                                {"tname": table_name},
                            )
                        ).scalar_one()
                    )
                else:
                    # dbstat 虚表可能未编译进 SQLite；逐表兜底降级为估算。
                    try:
                        dbstat_size = (
                            await session.execute(
                                text(
                                    "SELECT SUM(pgsize) FROM dbstat WHERE name = :tname"
                                ),
                                {"tname": table_name},
                            )
                        ).scalar_one()
                        size_bytes = int(dbstat_size or 0)
                    except Exception:
                        size_bytes = 0
                        approximate = True
                tables.append(
                    TableSpaceStat(
                        table=table_name,
                        bytes=size_bytes,
                        rows=int(row_count),
                        approximate=approximate,
                    )
                )

        data_files: tuple[DataFileStat, ...] = ()
        data_dir: str | None = None
        data_dir_total = 0
        # 文件系统统计绝不让 storage() 抛错：任何异常都降级为空结果。
        try:
            if self._data_dir:
                root = Path(self._data_dir)
                if root.is_dir():
                    data_dir = str(root)
                    stats = sorted(
                        (
                            DataFileStat(name=entry.name, bytes=_entry_size_bytes(entry))
                            for entry in root.iterdir()
                        ),
                        key=lambda f: f.bytes,
                        reverse=True,
                    )
                    data_files = tuple(stats)
                    data_dir_total = sum(f.bytes for f in stats)
        except OSError:
            data_files = ()
            data_dir_total = 0

        # 向量库统计走注入的 provider（保持本模块纯 SQL）；任何异常降级为 unavailable，
        # 绝不让 storage() 抛错。provider 未装配时 vector=None。
        vector: VectorStoreStat | None = None
        if self._vector_stats is not None:
            try:
                vector = await self._vector_stats()
            except Exception as exc:
                vector = VectorStoreStat(mode="unavailable", error=str(exc)[:200])

        return StorageReport(
            dialect=dialect,
            total_table_bytes=sum(t.bytes for t in tables),
            tables=tuple(tables),
            data_dir=data_dir,
            data_files=data_files,
            data_dir_total_bytes=data_dir_total,
            vector=vector,
        )
