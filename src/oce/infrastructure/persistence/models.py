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


class EmbeddingCredentialModel(Base):
    """扁平凭据：一行 = 一个账号，自带 embed/rerank 渠道配置（无独立 provider 表）。

    embed_endpoint/embed_model 为空则该行不参与嵌入解析；rerank_endpoint/rerank_model
    为空则不参与重排解析。resolve 时按 status='active' + priority 取最高优先级一条。
    """

    __tablename__ = "embedding_credentials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider = Column(String(64))  # 渠道标签（如 siliconflow），仅用于分组/复制
    name = Column(String(128), nullable=False)
    api_key = Column(String(512), nullable=False)
    api_key_hash = Column(String(64), nullable=False, unique=True)
    status = Column(String(16), nullable=False, default="active")
    priority = Column(Integer, nullable=False, default=100)

    embed_endpoint = Column(String(512))
    embed_model = Column(String(128))
    dimensions = Column(Integer, nullable=False, default=1024)
    max_batch_size = Column(Integer, nullable=False, default=32)
    max_batch_chars = Column(Integer, nullable=False, default=32_000)
    max_input_chars = Column(Integer, nullable=False, default=8_000)
    input_overlap_chars = Column(Integer, nullable=False, default=400)

    rerank_endpoint = Column(String(512))
    rerank_model = Column(String(128))

    timeout_seconds = Column(Integer, nullable=False, default=30)
    rate_limit = Column(Integer)
    note = Column(Text)
    last_used_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_embedding_credentials_status", "status"),
        Index("idx_embedding_credentials_priority", "priority"),
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
