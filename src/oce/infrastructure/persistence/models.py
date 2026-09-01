"""PostgreSQL/SQLite metadata models for blobs, chunks, and checkpoints."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

from oce.shared.database.session import Base


class ModelCredentialModel(Base):
    """多用途模型凭据：一行 = 一个 (kind, 账号) 通道。

    kind ∈ embed | rerank | llm_rerank | query_rewrite | intent。同一把 key 可服务多个
    用途/模型：唯一约束是 (kind, model, api_key_hash)，故同 key 跨 kind、同 kind 下同 key
    挂不同 model 都允许，只挡住 kind+model+key 三者全同的纯重复行。endpoint 语义随 kind 变化：
    embed/rerank 存完整 URL（/v1/embeddings、/v1/rerank），chat 三类（llm_rerank/
    query_rewrite/intent）存 base_url（/v1）。resolve 时按 kind + status='active' +
    priority 取最高优先级一条，取不到回落各自的环境变量。kind 专属参数列对其它 kind 恒为
    NULL，解析时缺失的字段回落 fallback 设置。
    """

    __tablename__ = "model_credentials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    kind = Column(String(16), nullable=False)
    provider = Column(String(64))  # 渠道标签（如 siliconflow），仅用于分组/复制
    name = Column(String(128), nullable=False)
    api_key = Column(String(512), nullable=False)
    api_key_hash = Column(String(64), nullable=False)
    endpoint = Column(String(512))
    model = Column(String(128))
    status = Column(String(16), nullable=False, default="active")
    priority = Column(Integer, nullable=False, default=100)
    timeout_seconds = Column(Integer, nullable=False, default=30)
    rate_limit = Column(Integer)
    note = Column(Text)

    # embed 专属
    dimensions = Column(Integer)
    max_batch_size = Column(Integer)
    max_batch_chars = Column(Integer)
    max_input_chars = Column(Integer)
    input_overlap_chars = Column(Integer)

    # rerank(API) 专属
    top_n = Column(Integer)
    min_score = Column(Float)

    # chat 三类专属：llm_rerank / query_rewrite / intent
    tpm_limit = Column(Integer)
    max_candidates = Column(Integer)
    output_top_k = Column(Integer)
    snippet_chars = Column(Integer)
    num_rewrites = Column(Integer)

    last_used_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "kind",
            "model",
            "api_key_hash",
            name="uq_model_credentials_kind_model_key",
        ),
        Index(
            "idx_model_credentials_kind_status_priority",
            "kind",
            "status",
            "priority",
        ),
    )


class BlobModel(Base):
    __tablename__ = "blobs"

    blob_name = Column(String(64), primary_key=True)
    path = Column(String(1024), nullable=False)
    content_size = Column(Integer, nullable=False)
    language = Column(String(32))
    file_type = Column(String(16), nullable=False, default="text")
    status = Column(String(16), nullable=False, default="pending")
    retry_count = Column(Integer, nullable=False, server_default="0")
    last_seen = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    error_message = Column(Text)

    __table_args__ = (
        Index("ix_blobs_status", "status"),
        Index("ix_blobs_last_seen", "last_seen"),
        Index("ix_blobs_language", "language"),
        Index("ix_blobs_retry_count", "retry_count"),
    )


class BlobStagingModel(Base):
    __tablename__ = "blob_staging"

    blob_name = Column(
        String(64),
        ForeignKey("blobs.blob_name", ondelete="CASCADE"),
        primary_key=True,
    )
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (Index("ix_blob_staging_created_at", "created_at"),)


class ChunkModel(Base):
    __tablename__ = "chunks"

    content_hash = Column(String(64), primary_key=True)
    content = Column(Text, nullable=False)
    content_size = Column(Integer, nullable=False)
    chunk_type = Column(String(32))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    embedded = Column(Boolean, server_default="false", nullable=False)

    __table_args__ = (
        Index("ix_chunks_chunk_type", "chunk_type"),
        Index("ix_chunks_embedded", "embedded"),
    )


class BlobChunkModel(Base):
    __tablename__ = "blob_chunks"

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    blob_name = Column(
        String(64),
        ForeignKey("blobs.blob_name", ondelete="CASCADE"),
        nullable=False,
    )
    content_hash = Column(
        String(64),
        ForeignKey("chunks.content_hash", ondelete="CASCADE"),
        nullable=False,
    )
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "blob_name",
            "content_hash",
            "start_line",
            "end_line",
            name="uq_blob_chunks_span",
        ),
        Index("ix_blob_chunks_blob_name", "blob_name"),
        Index("ix_blob_chunks_content_hash", "content_hash"),
        Index("ix_blob_chunks_blob_index", "blob_name", "chunk_index"),
    )


class ChainModel(Base):
    __tablename__ = "chains"

    chain_id = Column(String(64), primary_key=True)
    version = Column(Integer, nullable=False, default=1)
    description = Column(String(512))
    total_blobs = Column(Integer, nullable=False, default=0)
    total_chunks = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    __table_args__ = (Index("ix_chains_updated_at", "updated_at"),)


class ChainMemberModel(Base):
    __tablename__ = "chain_members"

    chain_id = Column(
        String(64),
        ForeignKey("chains.chain_id", ondelete="CASCADE"),
        primary_key=True,
    )
    blob_name = Column(String(64), primary_key=True)

    __table_args__ = (Index("ix_chain_members_blob_name", "blob_name"),)


class SymbolOccurrenceModel(Base):
    __tablename__ = "symbol_occurrences"

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    identifier = Column(String(256), nullable=False)
    blob_name = Column(
        String(64),
        ForeignKey("blobs.blob_name", ondelete="CASCADE"),
        nullable=False,
    )
    content_hash = Column(
        String(64),
        ForeignKey("chunks.content_hash", ondelete="CASCADE"),
        nullable=False,
    )
    kind = Column(String(16), nullable=False)
    start_line = Column(Integer, nullable=False)
    end_line = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_so_identifier", "identifier"),
        Index("idx_so_blob_name", "blob_name"),
        Index("idx_so_identifier_kind", "identifier", "kind"),
        Index("idx_so_content_hash", "content_hash"),
        UniqueConstraint("identifier", "blob_name", "content_hash", "kind", name="uq_symbol_occurrences_key"),
    )


class ApiCallMetricModel(Base):
    """每次 HTTP 请求一行：调用次数/耗时/状态码的事件明细。"""

    __tablename__ = "api_call_metrics"

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    ts = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    endpoint = Column(String(128), nullable=False)
    method = Column(String(8), nullable=False)
    status_code = Column(Integer, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    error_type = Column(String(64))

    __table_args__ = (
        Index("ix_api_call_metrics_ts", "ts"),
        Index("ix_api_call_metrics_endpoint", "endpoint"),
    )


class TokenUsageMetricModel(Base):
    """每次外部模型调用一行：embed/rerank/rewrite/intent 的 token 消耗明细。"""

    __tablename__ = "token_usage_metrics"

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    ts = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    kind = Column(String(16), nullable=False)
    model = Column(String(128), nullable=False)
    credential_id = Column(Integer)
    prompt_tokens = Column(Integer, nullable=False, server_default="0")
    completion_tokens = Column(Integer, nullable=False, server_default="0")
    total_tokens = Column(Integer, nullable=False, server_default="0")

    __table_args__ = (
        Index("ix_token_usage_metrics_ts", "ts"),
        Index("ix_token_usage_metrics_kind", "kind"),
        Index("ix_token_usage_metrics_credential_id", "credential_id"),
    )


class ResourceSampleModel(Base):
    """周期采样一行：磁盘/内存/CPU 的瞬时占用。"""

    __tablename__ = "resource_samples"

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    ts = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    disk_data_bytes = Column(BigInteger, nullable=False)
    disk_free_bytes = Column(BigInteger, nullable=False)
    disk_total_bytes = Column(BigInteger, nullable=False)
    mem_rss_bytes = Column(BigInteger, nullable=False)
    mem_percent = Column(Float, nullable=False)
    cpu_percent = Column(Float, nullable=False)

    __table_args__ = (Index("ix_resource_samples_ts", "ts"),)


class RetrievalMetricModel(Base):
    """一次检索的阶段耗时与结果审计。hit_count=0 即空回；source 区分真实检索与 overview 子查询。"""

    __tablename__ = "retrieval_metrics"

    id = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    ts = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    source = Column(String(32), nullable=False)
    scope_size = Column(Integer, nullable=True)
    hit_count = Column(Integer, nullable=False)
    total_ms = Column(Integer, nullable=False)
    intent = Column(String(32), nullable=True)
    path_boosted = Column(Boolean, nullable=False, server_default="false")
    query_text = Column(Text, nullable=True)
    intent_ms = Column(Integer, nullable=True)
    rewrite_ms = Column(Integer, nullable=True)
    dense_ms = Column(Integer, nullable=True)
    exact_ms = Column(Integer, nullable=True)
    fuse_ms = Column(Integer, nullable=True)
    rerank_ms = Column(Integer, nullable=True)
    llm_rerank_ms = Column(Integer, nullable=True)
    select_ms = Column(Integer, nullable=True)

    __table_args__ = (
        Index("ix_retrieval_metrics_ts", "ts"),
        Index("ix_retrieval_metrics_source", "source"),
        Index("ix_retrieval_metrics_hit_count", "hit_count"),
    )
