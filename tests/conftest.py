"""Shared domain factories and lightweight database fixtures."""

from __future__ import annotations

import hashlib
import os
import uuid

import pytest


def make_chunk_ref(
    *,
    content_hash: str | None = None,
    start_line: int = 1,
    end_line: int = 10,
):
    from oce.domain.chunk import ChunkRef

    value = content_hash or hashlib.sha256(f"test-{uuid.uuid4()}".encode()).hexdigest()
    return ChunkRef(value, start_line, end_line)


def make_blob(
    *,
    blob_name: str | None = None,
    path: str | None = None,
    status: str = "pending",
    chunks: list | None = None,
):
    from oce.domain.blob.blob import Blob, BlobStatus

    return Blob(
        blob_name=blob_name
        or hashlib.sha256(f"test-blob-{uuid.uuid4()}".encode()).hexdigest(),
        path=path or f"src/test_{uuid.uuid4().hex[:8]}.py",
        status=BlobStatus(status),
        chunks=chunks or [],
    )


def make_chain(
    *,
    chain_id: str | None = None,
    version: int = 1,
    members: list[str] | None = None,
):
    from oce.domain.chain.chain import Chain

    return Chain(
        chain_id=chain_id or str(uuid.uuid4()),
        version=version,
        members=set(members or []),
    )


def make_sha256(text: str = "") -> str:
    value = text or f"test-{uuid.uuid4()}"
    return hashlib.sha256(value.encode()).hexdigest()


@pytest.fixture
def freezed_time():
    from datetime import datetime, timezone

    from freezegun import freeze_time

    with freeze_time("2026-08-06 14:00:00"):
        yield datetime(2026, 8, 6, 14, tzinfo=timezone.utc)


@pytest.fixture(scope="session")
def test_db_url() -> str:
    return os.getenv("TEST_DATABASE_URL", "sqlite+aiosqlite:///:memory:")


@pytest.fixture(scope="session")
async def test_engine(test_db_url):
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(test_db_url, echo=False)
    yield engine
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    factory = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with factory() as session:
        yield session
        await session.rollback()
