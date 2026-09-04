"""ACE 兼容 HTTP DTO。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class FindMissingRequest(BaseModel):
    mem_object_names: list[str] = Field(default_factory=list)


class FindMissingResponse(BaseModel):
    unknown_memory_names: list[str] = Field(default_factory=list)
    nonindexed_blob_names: list[str] = Field(default_factory=list)


class BlobInput(BaseModel):
    content: str
    path: str = Field(min_length=1)


class BatchUploadRequest(BaseModel):
    blobs: list[BlobInput] = Field(default_factory=list)
    checkpoint_id: str = ""

    @field_validator("checkpoint_id", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: Any) -> Any:
        return "" if value is None else value


class BatchUploadResponse(BaseModel):
    blob_names: list[str] = Field(default_factory=list)


class ReloadCredentialsResponse(BaseModel):
    reloaded: bool
    pool_size: int = 0
    reason: str | None = None


class BlobsPayload(BaseModel):
    checkpoint_id: str = ""
    added_blobs: list[str] = Field(default_factory=list)
    deleted_blobs: list[str] = Field(default_factory=list)

    @field_validator("checkpoint_id", mode="before")
    @classmethod
    def none_to_empty_string(cls, value: Any) -> Any:
        return "" if value is None else value

    @field_validator("added_blobs", "deleted_blobs", mode="before")
    @classmethod
    def none_to_empty_list(cls, value: Any) -> Any:
        return [] if value is None else value


class CodebaseRetrievalRequest(BaseModel):
    information_request: str = Field(min_length=1)
    blobs: BlobsPayload = Field(default_factory=BlobsPayload)
    chat_history: list[Any] = Field(default_factory=list)


class CodebaseRetrievalResponse(BaseModel):
    formatted_retrieval: str
    codebase_retrieval_elapsed_ms: int


class CheckpointBlobsRequest(BaseModel):
    blobs: BlobsPayload = Field(default_factory=BlobsPayload)


class CheckpointBlobsResponse(BaseModel):
    new_checkpoint_id: str


class BlobStatusRequest(BaseModel):
    blobs: BlobsPayload = Field(default_factory=BlobsPayload)


class BlobStatusResponse(BaseModel):
    unknown_blob_names: list[str] = Field(default_factory=list)
    nonindexed_blob_names: list[str] = Field(default_factory=list)
    checkpoint_not_found: bool = False


class ApiCallStatsResponse(BaseModel):
    count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: int = 0
    p95_latency_ms: int = 0
    max_latency_ms: int = 0


class TokenKindStatsResponse(BaseModel):
    kind: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class RetrievalStatsResponse(BaseModel):
    count: int = 0
    empty_count: int = 0
    empty_rate: float = 0.0


class ResourceSnapshotResponse(BaseModel):
    ts: datetime | None = None
    mem_rss_bytes: int = 0
    mem_percent: float = 0.0
    cpu_percent: float = 0.0
    disk_free_bytes: int = 0
    disk_total_bytes: int = 0
    disk_data_bytes: int = 0


class MonitoringStatsResponse(BaseModel):
    window_hours: int
    api_calls: ApiCallStatsResponse = Field(default_factory=ApiCallStatsResponse)
    tokens: list[TokenKindStatsResponse] = Field(default_factory=list)
    tokens_total: int = 0
    retrieval: RetrievalStatsResponse = Field(default_factory=RetrievalStatsResponse)
    resource: ResourceSnapshotResponse | None = None


# ---------------------------------------------------------------- 报表响应模型
# 字段与 shared.reports_read 的 dataclass 同名，便于 model_validate(from_attributes)。


class ApiCallBucketResponse(BaseModel):
    ts: datetime
    count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p50_latency_ms: int = 0
    p95_latency_ms: int = 0
    max_latency_ms: int = 0


class EndpointStatResponse(BaseModel):
    endpoint: str
    method: str
    count: int = 0
    error_count: int = 0
    error_rate: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: int = 0


class ErrorStatResponse(BaseModel):
    status_code: int
    error_type: str | None = None
    count: int = 0
    last_ts: datetime | None = None


class ApiCallsReportResponse(BaseModel):
    window_hours: int
    bucket: str
    buckets: list[ApiCallBucketResponse] = Field(default_factory=list)
    endpoints: list[EndpointStatResponse] = Field(default_factory=list)
    errors: list[ErrorStatResponse] = Field(default_factory=list)


class RetrievalBucketResponse(BaseModel):
    ts: datetime
    count: int = 0
    empty_count: int = 0
    empty_rate: float = 0.0
    avg_hit_count: float = 0.0
    avg_total_ms: float = 0.0
    p95_total_ms: int = 0


class StageStatResponse(BaseModel):
    stage: str
    count: int = 0
    avg_ms: float = 0.0
    p95_ms: int = 0
    max_ms: int = 0


class IntentStatResponse(BaseModel):
    intent: str | None = None
    count: int = 0
    empty_count: int = 0
    empty_rate: float = 0.0
    avg_total_ms: float = 0.0
    path_boosted_count: int = 0


class ScopeBucketStatResponse(BaseModel):
    label: str
    count: int = 0
    empty_rate: float = 0.0
    p95_total_ms: int = 0


class RetrievalReportResponse(BaseModel):
    window_hours: int
    bucket: str
    buckets: list[RetrievalBucketResponse] = Field(default_factory=list)
    stages: list[StageStatResponse] = Field(default_factory=list)
    intents: list[IntentStatResponse] = Field(default_factory=list)
    scopes: list[ScopeBucketStatResponse] = Field(default_factory=list)


class RetrievalQueryDetailResponse(BaseModel):
    ts: datetime
    source: str
    query_text: str | None = None
    total_ms: int
    hit_count: int
    scope_size: int | None = None
    intent: str | None = None
    path_boosted: bool = False


class RetrievalQueryListResponse(BaseModel):
    window_hours: int
    items: list[RetrievalQueryDetailResponse] = Field(default_factory=list)


class TokenBucketResponse(BaseModel):
    ts: datetime
    kind: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ModelTokenStatResponse(BaseModel):
    # 字段名 model 与 pydantic 保护前缀冲突，显式放开以保持与读模型同名
    model_config = ConfigDict(protected_namespaces=())

    model: str
    kind: str
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    avg_tokens_per_call: float = 0.0


class CredentialTokenStatResponse(BaseModel):
    credential_id: int | None = None
    calls: int = 0
    total_tokens: int = 0


class TokensReportResponse(BaseModel):
    window_hours: int
    bucket: str
    buckets: list[TokenBucketResponse] = Field(default_factory=list)
    models: list[ModelTokenStatResponse] = Field(default_factory=list)
    credentials: list[CredentialTokenStatResponse] = Field(default_factory=list)
    tokens_total: int = 0


class CountStatResponse(BaseModel):
    key: str
    count: int = 0


class IndexInventoryReportResponse(BaseModel):
    blob_total: int = 0
    blob_by_status: list[CountStatResponse] = Field(default_factory=list)
    blob_by_language: list[CountStatResponse] = Field(default_factory=list)
    blob_retrying: int = 0
    blob_content_bytes: int = 0
    chunk_total: int = 0
    chunk_pending_embed: int = 0
    chunk_by_type: list[CountStatResponse] = Field(default_factory=list)
    chunk_content_bytes: int = 0
    blob_chunk_links: int = 0
    symbol_total: int = 0
    symbol_by_kind: list[CountStatResponse] = Field(default_factory=list)
    chain_total: int = 0
    chain_stale_7d: int = 0
    chain_stale_30d: int = 0
    staging_rows: int = 0


class ResourceBucketResponse(BaseModel):
    ts: datetime
    avg_cpu_percent: float = 0.0
    max_cpu_percent: float = 0.0
    avg_mem_percent: float = 0.0
    max_mem_rss_bytes: int = 0
    disk_data_bytes: int = 0
    disk_free_bytes: int = 0


class ResourcesReportResponse(BaseModel):
    window_hours: int
    bucket: str
    buckets: list[ResourceBucketResponse] = Field(default_factory=list)
    disk_total_bytes: int = 0
    disk_growth_bytes_per_day: float = 0.0
    disk_days_until_full: float | None = None


class TableSpaceStatResponse(BaseModel):
    table: str
    bytes: int = 0
    rows: int = 0
    approximate: bool = False


class DataFileStatResponse(BaseModel):
    name: str
    bytes: int = 0


class VectorCollectionStatResponse(BaseModel):
    name: str
    rows: int = 0
    est_bytes: int = 0


class VectorStoreStatResponse(BaseModel):
    mode: str = "unavailable"
    collections: list[VectorCollectionStatResponse] = Field(default_factory=list)
    file_bytes: int = 0
    error: str | None = None


class StorageReportResponse(BaseModel):
    dialect: str = ""
    total_table_bytes: int = 0
    tables: list[TableSpaceStatResponse] = Field(default_factory=list)
    data_dir: str | None = None
    data_files: list[DataFileStatResponse] = Field(default_factory=list)
    data_dir_total_bytes: int = 0
    vector: VectorStoreStatResponse | None = None


# 凭据用途：embed/rerank 走 REST（/v1/embeddings、/v1/rerank）；后三类走 chat。
CredentialKind = Literal[
    "embed", "rerank", "llm_rerank", "query_rewrite", "intent"
]


class CredentialResponse(BaseModel):
    """凭据视图：脱敏，只暴露 api_key 尾 4 位。kind 专属参数对其它 kind 为 None。"""

    id: int
    kind: CredentialKind
    provider: str | None = None
    name: str
    status: str
    priority: int
    endpoint: str | None = None
    model: str | None = None
    timeout_seconds: int
    rate_limit: int | None = None
    note: str | None = None
    dimensions: int | None = None
    max_batch_size: int | None = None
    max_batch_chars: int | None = None
    max_input_chars: int | None = None
    input_overlap_chars: int | None = None
    top_n: int | None = None
    min_score: float | None = None
    tpm_limit: int | None = None
    max_candidates: int | None = None
    output_top_k: int | None = None
    snippet_chars: int | None = None
    num_rewrites: int | None = None
    api_key_last4: str
    last_used_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CredentialListResponse(BaseModel):
    credentials: list[CredentialResponse] = Field(default_factory=list)


class CredentialCreateRequest(BaseModel):
    kind: CredentialKind
    name: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    provider: str | None = None
    status: Literal["active", "disabled"] = "active"
    priority: int = 100
    endpoint: str | None = None
    model: str | None = None
    timeout_seconds: int = 30
    rate_limit: int | None = None
    note: str | None = None
    dimensions: int | None = None
    max_batch_size: int | None = None
    max_batch_chars: int | None = None
    max_input_chars: int | None = None
    input_overlap_chars: int | None = None
    top_n: int | None = None
    min_score: float | None = None
    tpm_limit: int | None = None
    max_candidates: int | None = None
    output_top_k: int | None = None
    snippet_chars: int | None = None
    num_rewrites: int | None = None


class CredentialUpdateRequest(BaseModel):
    """部分更新：省略的字段不改；api_key 提供则同步刷新 hash。"""

    kind: CredentialKind | None = None
    name: str | None = None
    api_key: str | None = None
    provider: str | None = None
    status: Literal["active", "disabled"] | None = None
    priority: int | None = None
    endpoint: str | None = None
    model: str | None = None
    timeout_seconds: int | None = None
    rate_limit: int | None = None
    note: str | None = None
    dimensions: int | None = None
    max_batch_size: int | None = None
    max_batch_chars: int | None = None
    max_input_chars: int | None = None
    input_overlap_chars: int | None = None
    top_n: int | None = None
    min_score: float | None = None
    tpm_limit: int | None = None
    max_candidates: int | None = None
    output_top_k: int | None = None
    snippet_chars: int | None = None
    num_rewrites: int | None = None


class CredentialDuplicateRequest(BaseModel):
    """从源凭据克隆一个新通道：所有字段可选，提供即覆盖，省略即继承源行。

    省略 api_key 即复用源 key，配合覆盖 kind/model 可把某把 key 的通道复制成别的用途
    （如复制 embed 行改成 rerank），不再撞唯一约束。
    """

    name: str | None = Field(default=None, min_length=1)
    api_key: str | None = Field(default=None, min_length=1)
    kind: CredentialKind | None = None
    provider: str | None = None
    status: Literal["active", "disabled"] | None = None
    priority: int | None = None
    endpoint: str | None = None
    model: str | None = None
    timeout_seconds: int | None = None
    rate_limit: int | None = None
    note: str | None = None
    dimensions: int | None = None
    max_batch_size: int | None = None
    max_batch_chars: int | None = None
    max_input_chars: int | None = None
    input_overlap_chars: int | None = None
    top_n: int | None = None
    min_score: float | None = None
    tpm_limit: int | None = None
    max_candidates: int | None = None
    output_top_k: int | None = None
    snippet_chars: int | None = None
    num_rewrites: int | None = None


class QueueStatusResponse(BaseModel):
    enabled: bool
    main_size: int = 0
    inflight: int = 0
    db_pending: int = 0


class QueueResetRequest(BaseModel):
    mode: Literal["sync", "purge"] = "sync"
    requeue: bool = True


class QueueResetResponse(BaseModel):
    removed: int = 0
    requeued: int = 0
    queue_size: int = 0
    db_pending: int = 0


class RequeueStaleRequest(BaseModel):
    stale_hours: int = 24
    limit: int = 100


class RequeueStaleResponse(BaseModel):
    requeued_count: int = 0


class GcRequest(BaseModel):
    ttl_days: int = 30
    dry_run: bool = True
    limit: int = 1000


class GcResponse(BaseModel):
    dry_run: bool
    ttl_days: int
    expired_chains: int = 0
    expired_blobs: int = 0
    deletable_blobs: int = 0
    skipped_inflight: int = 0
    deleted_chains: int = 0
    deleted_blobs: int = 0
