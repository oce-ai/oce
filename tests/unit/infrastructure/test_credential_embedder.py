"""Database-backed embedding credential resolution tests."""

from __future__ import annotations

import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oce.shared.database.session import Base
from oce.infrastructure.embed.credential_embedder import CredentialConfiguredEmbedder
from oce.infrastructure.persistence.models import EmbeddingCredentialModel, EmbeddingProviderModel
from oce.shared.config.settings import EmbeddingSettings


async def _runtime():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def test_active_credential_overrides_provider_batch_defaults():
    engine, sessions = await _runtime()
    async with sessions() as session:
        provider = EmbeddingProviderModel(
            code="siliconflow",
            display_name="SiliconFlow",
            embed_endpoint="https://example.test/v1/embeddings",
            embed_model="embedding-model",
            dimensions=1024,
            max_batch_size=32,
            max_batch_chars=32_000,
        )
        session.add(provider)
        await session.flush()
        session.add(
            EmbeddingCredentialModel(
                provider_id=provider.id,
                name="primary",
                api_key="database-key",
                api_key_hash="hash",
                priority=10,
                max_batch_size=8,
                max_batch_chars=24_000,
            )
        )
        await session.commit()

    embedder = CredentialConfiguredEmbedder(
        sessions,
        EmbeddingSettings(api_key="fallback-key"),
        expected_dimensions=1024,
    )
    config = await embedder._resolve_config()

    assert config.api_key == "database-key"
    assert config.max_batch_size == 8
    assert config.max_batch_chars == 24_000
    await engine.dispose()


async def test_environment_settings_are_used_without_active_credential():
    engine, sessions = await _runtime()
    settings = EmbeddingSettings(
        api_key="fallback-key",
        max_batch_size=7,
        max_batch_chars=31_000,
    )
    embedder = CredentialConfiguredEmbedder(
        sessions,
        settings,
        expected_dimensions=1024,
    )

    config = await embedder._resolve_config()

    assert config.api_key == "fallback-key"
    assert config.max_batch_size == 7
    assert config.max_batch_chars == 31_000
    await engine.dispose()


async def test_reload_waits_for_inflight_request_before_closing_old_client():
    engine, sessions = await _runtime()
    embedder = CredentialConfiguredEmbedder(
        sessions,
        EmbeddingSettings(api_key="fallback-key"),
        expected_dimensions=1024,
    )
    started = asyncio.Event()
    release = asyncio.Event()

    class Delegate:
        def __init__(self, blocking: bool = False) -> None:
            self.blocking = blocking
            self.closed = False

        async def embed_documents(self, _texts):
            started.set()
            if self.blocking:
                await release.wait()
            return [[1.0]]

        async def close(self):
            self.closed = True

    old = Delegate(blocking=True)
    replacement = Delegate()
    embedder._delegate = old

    async def resolve_config():
        return object()

    embedder._resolve_config = resolve_config
    embedder._build_delegate = lambda _config: replacement

    request = asyncio.create_task(embedder.embed_documents(["source"]))
    await started.wait()
    await embedder.reload()

    assert old.closed is False
    assert embedder._delegate is replacement

    release.set()
    assert await request == [[1.0]]
    assert old.closed is True
    await embedder.close()
    await engine.dispose()
