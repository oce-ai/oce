"""报表只读聚合契约：/admin/reports/* 的读模型 DTO 与 reader 端口。

与 metrics_read.py 同构：本模块只定义读出聚合的结果结构与 reader Protocol；
infra 实现 SQL 聚合、application 编排、api 映射 DTO——三层都只依赖这里的纯数据结构。

约定：
- 时间序列报表按 ``bucket``（"hour" | "day"）分桶，窗口由 ``window_hours`` 界定；
- 分桶与分位数在 Python 侧计算，SQL 只做窗口过滤/基础聚合，保证 SQLite/PG 可移植；
- 报表是旁路读路径，绝不写库。
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

# ---------------------------------------------------------------- API 健康报表


@dataclass(frozen=True)
class ApiCallBucket:
    ts: datetime
    count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: int = 0
    p95_latency_ms: int = 0
    max_latency_ms: int = 0


@dataclass(frozen=True)
class EndpointStat:
    endpoint: str
    method: str
    count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: int = 0


@dataclass(frozen=True)
class ErrorStat:
    status_code: int
    error_type: str | None = None
    count: int = 0
    last_ts: datetime | None = None


@dataclass(frozen=True)
class ApiCallsReport:
    window_hours: int
    bucket: str
    buckets: tuple[ApiCallBucket, ...] = ()
    endpoints: tuple[EndpointStat, ...] = ()
    errors: tuple[ErrorStat, ...] = ()


# ---------------------------------------------------------------- 检索质量报表


@dataclass(frozen=True)
class RetrievalBucket:
    ts: datetime
    count: int = 0
    empty_count: int = 0
    empty_rate: float = 0.0
    avg_hit_count: float = 0.0
    avg_total_ms: float = 0.0
    p95_total_ms: int = 0


@dataclass(frozen=True)
class StageStat:
    """检索管线单阶段耗时统计；stage 取 retrieval_metrics 的 *_ms 列名前缀。"""

    stage: str
    count: int = 0
    avg_ms: float = 0.0
    p95_ms: int = 0
    max_ms: int = 0


@dataclass(frozen=True)
class IntentStat:
    intent: str | None
    count: int = 0
    empty_count: int = 0
    empty_rate: float = 0.0
    avg_total_ms: float = 0.0
    path_boosted_count: int = 0


@dataclass(frozen=True)
class ScopeBucketStat:
    """按工作集规模分桶：label 如 "1-100"、"101-1000"、">10000"。"""

    label: str
    count: int = 0
    empty_rate: float = 0.0
    p95_total_ms: int = 0


@dataclass(frozen=True)
class RetrievalReport:
    window_hours: int
    bucket: str
    buckets: tuple[RetrievalBucket, ...] = ()
    stages: tuple[StageStat, ...] = ()
    intents: tuple[IntentStat, ...] = ()
    scopes: tuple[ScopeBucketStat, ...] = ()


@dataclass(frozen=True)
class RetrievalQueryDetail:
    """慢查询 / 空回查询明细行。query_text 可能因配置关闭而为 None。"""

    ts: datetime
    source: str
    query_text: str | None
    total_ms: int
    hit_count: int
    scope_size: int | None = None
    intent: str | None = None
    path_boosted: bool = False


# ---------------------------------------------------------------- Token 用量报表


@dataclass(frozen=True)
class TokenBucket:
    ts: datetime
    kind: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class ModelTokenStat:
    model: str
    kind: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    avg_tokens_per_call: float = 0.0


@dataclass(frozen=True)
class CredentialTokenStat:
    """credential_id 为 None 表示回退到了环境变量凭据。"""

    credential_id: int | None
    calls: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class TokensReport:
    window_hours: int
    bucket: str
    buckets: tuple[TokenBucket, ...] = ()
    models: tuple[ModelTokenStat, ...] = ()
    credentials: tuple[CredentialTokenStat, ...] = ()
    tokens_total: int = 0


# ---------------------------------------------------------------- 索引资产报表


@dataclass(frozen=True)
class CountStat:
    """通用 (key, count) 分布行，如按 status/language/chunk_type/kind 分组。"""

    key: str
    count: int = 0


@dataclass(frozen=True)
class IndexInventoryReport:
    """当前快照统计，无时间窗口。"""

    blob_total: int = 0
    blob_by_status: tuple[CountStat, ...] = ()
    blob_by_language: tuple[CountStat, ...] = ()
    blob_retrying: int = 0
    blob_content_bytes: int = 0
    chunk_total: int = 0
    chunk_pending_embed: int = 0
    chunk_by_type: tuple[CountStat, ...] = ()
    chunk_content_bytes: int = 0
    blob_chunk_links: int = 0
    symbol_total: int = 0
    symbol_by_kind: tuple[CountStat, ...] = ()
    chain_total: int = 0
    chain_stale_7d: int = 0
    chain_stale_30d: int = 0
    staging_rows: int = 0


# ---------------------------------------------------------------- 资源容量报表


@dataclass(frozen=True)
class ResourceBucket:
    ts: datetime
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_mem_percent: float = 0.0
    max_mem_rss_bytes: int = 0
    disk_data_bytes: int = 0
    disk_free_bytes: int = 0


@dataclass(frozen=True)
class ResourcesReport:
    window_hours: int
    bucket: str
    buckets: tuple[ResourceBucket, ...] = ()
    disk_total_bytes: int = 0
    disk_growth_bytes_per_day: float = 0.0
    disk_days_until_full: float | None = None


# ---------------------------------------------------------------- 数据空间占用报表


@dataclass(frozen=True)
class TableSpaceStat:
    """单表空间占用。PG 用 pg_total_relation_size；SQLite 用 dbstat（不可用时
    回退为按行数近似），因此 approximate=True 表示估算值。"""

    table: str
    bytes: int = 0
    rows: int = 0
    approximate: bool = False


@dataclass(frozen=True)
class DataFileStat:
    """数据目录内单文件/子目录占用（个人模式：SQLite db、Milvus Lite 文件等）。"""

    name: str
    bytes: int = 0


@dataclass(frozen=True)
class VectorCollectionStat:
    """单个 Milvus collection 的占用。rows 精确；est_bytes 为估算
    （rows x dense_dim x 4 字节的 dense 向量体积，不含标量字段与索引开销）。"""

    name: str
    rows: int = 0
    est_bytes: int = 0


@dataclass(frozen=True)
class VectorStoreStat:
    """向量库占用。mode: "lite"（本地文件，file_bytes 为真实尺寸）| "server"
    （远程 Milvus，无法取磁盘字节，仅行数 + 估算）| "unavailable"（连不上/未装配）。"""

    mode: str = "unavailable"
    collections: tuple[VectorCollectionStat, ...] = ()
    file_bytes: int = 0
    error: str | None = None


@dataclass(frozen=True)
class StorageReport:
    dialect: str = ""
    total_table_bytes: int = 0
    tables: tuple[TableSpaceStat, ...] = ()
    data_dir: str | None = None
    data_files: tuple[DataFileStat, ...] = ()
    data_dir_total_bytes: int = 0
    vector: VectorStoreStat | None = None


# ---------------------------------------------------------------- reader 端口


class ReportsReader(Protocol):
    """报表聚合读端口；infra 用 SQL + 文件系统实现。"""

    async def api_calls(self, window_hours: int, bucket: str) -> ApiCallsReport: ...

    async def retrieval(self, window_hours: int, bucket: str) -> RetrievalReport: ...

    async def slow_queries(
        self, window_hours: int, limit: int
    ) -> tuple[RetrievalQueryDetail, ...]: ...

    async def empty_queries(
        self, window_hours: int, limit: int
    ) -> tuple[RetrievalQueryDetail, ...]: ...

    async def tokens(self, window_hours: int, bucket: str) -> TokensReport: ...

    async def index_inventory(self) -> IndexInventoryReport: ...

    async def resources(self, window_hours: int, bucket: str) -> ResourcesReport: ...

    async def storage(self) -> StorageReport: ...
