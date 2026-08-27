"""Database-backed rerank credential resolution tests."""

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oce.shared.database.session import Base
from oce.infrastructure.embed.credential_reranker import CredentialConfiguredReranker
from oce.infrastructure.persistence.models import EmbeddingCredentialModel, EmbeddingProviderModel
from oce.shared.config.settings import RerankSettings


async def _runtime():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def test_active_credential_configures_reranker():
    engine, sessions = await _runtime()
    async with sessions() as session:
        provider = EmbeddingProviderModel(
            code="siliconflow",
            display_name="SiliconFlow",
            rerank_endpoint="https://database.test/v1/rerank",
            rerank_model="database-reranker",
        )
        session.add(provider)
        await session.flush()
        session.add(
            EmbeddingCredentialModel(
                provider_id=provider.id,
                name="primary",
                api_key="database-key",
                api_key_hash="rerank-hash",
                priority=1,
                timeout_seconds=45,
            )
        )
        await session.commit()

    reranker = CredentialConfiguredReranker(
        sessions,
        RerankSettings(api_key="fallback-key", enabled=True),
        fallback_embedding_key=None,
    )
    config = await reranker._resolve_config()

    assert config is not None
    assert config.endpoint == "https://database.test/v1/rerank"
    assert config.api_key == "database-key"
    assert config.model == "database-reranker"
    assert config.timeout_seconds == 45
    await engine.dispose()


async def test_missing_rerank_key_uses_noop_delegate():
    engine, sessions = await _runtime()
    reranker = CredentialConfiguredReranker(
        sessions,
        RerankSettings(api_key=None, enabled=True),
        fallback_embedding_key=None,
    )

    assert await reranker._resolve_config() is None
    assert await reranker.rerank("query", ["hit"]) == ["hit"]
    await reranker.close()
    await engine.dispose()
