"""Database-backed embedding credential resolution tests."""

from __future__ import annotations

import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oce.shared.database.session import Base
from oce.infrastructure.embed.credential_embedder import CredentialConfiguredEmbedder
from oce.infrastructure.persistence.models import ModelCredentialModel
from oce.shared.config.settings import EmbeddingSettings


async def _runtime():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def test_active_credential_batch_settings_used():
    engine, sessions = await _runtime()
    async with sessions() as session:
        session.add(
            ModelCredentialModel(
                kind="embed",
                provider="siliconflow",
                name="primary",
                api_key="database-key",
                api_key_hash="hash",
                priority=10,
                endpoint="https://example.test/v1/embeddings",
                model="embedding-model",
                dimensions=1024,
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


async def test_credential_id_and_usage_callback_wired_through():
    """DB 凭证的 id 填入 config，并把 credential_id + on_usage 透传给底层 delegate。"""
    engine, sessions = await _runtime()
    async with sessions() as session:
        credential = ModelCredentialModel(
            kind="embed",
            provider="siliconflow",
            name="primary",
            api_key="database-key",
            api_key_hash="hash",
            priority=10,
            endpoint="https://example.test/v1/embeddings",
            model="embedding-model",
            dimensions=1024,
        )
        session.add(credential)
        await session.commit()
        credential_id = credential.id

    async def _cb(*_args):
        return None

    embedder = CredentialConfiguredEmbedder(
        sessions,
        EmbeddingSettings(api_key="fallback-key"),
        expected_dimensions=1024,
        on_usage=_cb,
    )
    config = await embedder._resolve_config()
    assert config.credential_id == credential_id

    delegate = embedder._build_delegate(config)
    assert delegate._credential_id == credential_id
    assert delegate._on_usage is _cb
    await delegate.close()
    await engine.dispose()
