"""ACE 兼容 HTTP DTO。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


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


class CredentialResponse(BaseModel):
    """凭据视图：脱敏，只暴露 api_key 尾 4 位。"""

    id: int
    provider: str | None = None
    name: str
    status: str
    priority: int
    embed_endpoint: str | None = None
    embed_model: str | None = None
    dimensions: int
    max_batch_size: int
    max_batch_chars: int
    max_input_chars: int
    input_overlap_chars: int
    rerank_endpoint: str | None = None
    rerank_model: str | None = None
    timeout_seconds: int
    rate_limit: int | None = None
    note: str | None = None
    api_key_last4: str
    last_used_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CredentialListResponse(BaseModel):
    credentials: list[CredentialResponse] = Field(default_factory=list)


class CredentialCreateRequest(BaseModel):
    name: str = Field(min_length=1)
    api_key: str = Field(min_length=1)
    provider: str | None = None
    status: Literal["active", "disabled"] = "active"
    priority: int = 100
    embed_endpoint: str | None = None
    embed_model: str | None = None
    dimensions: int = 1024
    max_batch_size: int = 32
    max_batch_chars: int = 32_000
    max_input_chars: int = 8_000
    input_overlap_chars: int = 400
    rerank_endpoint: str | None = None
    rerank_model: str | None = None
    timeout_seconds: int = 30
    rate_limit: int | None = None
    note: str | None = None


class CredentialUpdateRequest(BaseModel):
    """部分更新：省略的字段不改；api_key 提供则同步刷新 hash。"""

    name: str | None = None
    api_key: str | None = None
    provider: str | None = None
    status: Literal["active", "disabled"] | None = None
    priority: int | None = None
    embed_endpoint: str | None = None
    embed_model: str | None = None
    dimensions: int | None = None
    max_batch_size: int | None = None
    max_batch_chars: int | None = None
    max_input_chars: int | None = None
    input_overlap_chars: int | None = None
    rerank_endpoint: str | None = None
    rerank_model: str | None = None
    timeout_seconds: int | None = None
    rate_limit: int | None = None
    note: str | None = None


class CredentialDuplicateRequest(BaseModel):
    """复制源凭据的渠道配置，仅换 name + api_key。"""

    name: str = Field(min_length=1)
    api_key: str = Field(min_length=1)


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
