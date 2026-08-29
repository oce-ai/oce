"""Database-backed rerank credential resolution tests."""

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oce.shared.database.session import Base
from oce.infrastructure.embed.credential_reranker import CredentialConfiguredReranker
from oce.infrastructure.persistence.models import ModelCredentialModel
from oce.shared.config.settings import RerankSettings


async def _runtime():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def test_active_credential_configures_reranker():
    engine, sessions = await _runtime()
    async with sessions() as session:
        session.add(
            ModelCredentialModel(
                kind="rerank",
                provider="siliconflow",
                name="primary",
                api_key="database-key",
                api_key_hash="rerank-hash",
                priority=1,
                timeout_seconds=45,
                endpoint="https://database.test/v1/rerank",
                model="database-reranker",
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


async def test_credential_id_and_usage_callback_wired_through():
    """DB 凭证的 id 填入 config，并把 credential_id + on_usage 透传给底层 delegate。"""
    engine, sessions = await _runtime()
    async with sessions() as session:
        credential = ModelCredentialModel(
            kind="rerank",
            provider="siliconflow",
            name="primary",
            api_key="database-key",
            api_key_hash="rerank-hash",
            priority=1,
            timeout_seconds=45,
            endpoint="https://database.test/v1/rerank",
            model="database-reranker",
        )
        session.add(credential)
        await session.commit()
        credential_id = credential.id

    async def _cb(*_args):
        return None

    reranker = CredentialConfiguredReranker(
        sessions,
        RerankSettings(api_key="fallback-key", enabled=True),
        fallback_embedding_key=None,
        on_usage=_cb,
    )
    config = await reranker._resolve_config()
    assert config is not None
    assert config.credential_id == credential_id

    delegate = reranker._build_delegate(config)
    assert delegate._credential_id == credential_id
    assert delegate._on_usage is _cb
    await delegate.close()
    await engine.dispose()
