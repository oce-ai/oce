"""PostgreSQL/SQLite metadata models for blobs, chunks, and checkpoints."""

from __future__ import annotations

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)

from oce.shared.database.session import Base


class EmbeddingProviderModel(Base):
    __tablename__ = "embedding_providers"

    id = Column(Integer, primary_key=True, autoincrement=True)
    code = Column(String(64), nullable=False, unique=True)
    display_name = Column(String(128), nullable=False)
    embed_endpoint = Column(String(512))
    embed_model = Column(String(128))
    rerank_endpoint = Column(String(512))
    rerank_model = Column(String(128))
    dimensions = Column(Integer, nullable=False, default=1024)
    max_batch_size = Column(Integer, nullable=False, default=32)
    max_batch_chars = Column(Integer, nullable=False, default=32_000)
    max_input_chars = Column(Integer, nullable=False, default=8_000)
    input_overlap_chars = Column(Integer, nullable=False, default=400)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class EmbeddingCredentialModel(Base):
    __tablename__ = "embedding_credentials"

    id = Column(Integer, primary_key=True, autoincrement=True)
    provider_id = Column(
        Integer,
        ForeignKey("embedding_providers.id", ondelete="CASCADE"),
        nullable=False,
    )
    name = Column(String(128), nullable=False)
    api_key = Column(String(512), nullable=False)
    api_key_hash = Column(String(64), nullable=False, unique=True)
    priority = Column(Integer, nullable=False, default=100)
    status = Column(String(16), nullable=False, default="active")
    max_batch_size = Column(Integer)
    max_batch_chars = Column(Integer)
    note = Column(Text)
    rate_limit = Column(Integer)
    timeout_seconds = Column(Integer, nullable=False, default=30)
    last_used_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (
        Index("idx_embedding_credentials_provider_id", "provider_id"),
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
