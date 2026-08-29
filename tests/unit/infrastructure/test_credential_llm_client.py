"""按 kind 解析的 chat-LLM 凭证客户端测试。"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oce.infrastructure.llm.credential_llm_client import CredentialConfiguredLLMClient
from oce.infrastructure.persistence.models import ModelCredentialModel
from oce.shared.config.settings import LLMSettings
from oce.shared.database.session import Base


async def _runtime():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, async_sessionmaker(engine, expire_on_commit=False)


async def test_resolves_active_credential_for_kind():
    engine, sessions = await _runtime()
    async with sessions() as session:
        cred = ModelCredentialModel(
            kind="llm_rerank",
            name="rk",
            api_key="db-key",
            api_key_hash="h1",
            priority=5,
            endpoint="https://db.test/v1",
            model="db-model",
            tpm_limit=999,
            timeout_seconds=30,
        )
        session.add(cred)
        # 不同 kind 的行即使优先级更高也不应被 llm_rerank 选中
        session.add(
            ModelCredentialModel(
                kind="intent",
                name="i",
                api_key="other",
                api_key_hash="h2",
                priority=1,
                endpoint="https://x.test/v1",
                model="m",
            )
        )
        await session.commit()
        cid = cred.id

    client = CredentialConfiguredLLMClient(
        "llm_rerank",
        sessions,
        LLMSettings(api_key="env-key"),
        fallback_model="env-model",
    )
    config = await client._resolve_config()

    assert config.api_key == "db-key"
    assert config.base_url == "https://db.test/v1"
    assert config.model == "db-model"
    assert config.tpm_limit == 999
    assert config.credential_id == cid
    await engine.dispose()


async def test_falls_back_to_env_without_credential():
    engine, sessions = await _runtime()
    client = CredentialConfiguredLLMClient(
        "query_rewrite",
        sessions,
        LLMSettings(api_key="env-key", base_url="https://env.test/v1", tpm_limit=12_345),
        fallback_model="env-model",
    )
    config = await client._resolve_config()

    assert config.api_key == "env-key"
    assert config.base_url == "https://env.test/v1"
    assert config.model is None
    assert config.tpm_limit == 12_345
    assert config.credential_id == 0
    await engine.dispose()


async def test_build_delegate_wires_credential_id_and_usage():
    engine, sessions = await _runtime()

    async def _cb(*_args):
        return None

    client = CredentialConfiguredLLMClient(
        "intent",
        sessions,
        LLMSettings(api_key="env-key"),
        fallback_model="env-model",
        on_usage=_cb,
    )
    config = await client._resolve_config()
    delegate = client._build_delegate(config)

    assert delegate._credential_id == 0
    assert delegate._on_usage is _cb
    await engine.dispose()


async def test_chat_model_precedence():
    """凭证 model > 调用方 model > fallback_model。"""
    engine, sessions = await _runtime()
    async with sessions() as session:
        session.add(
            ModelCredentialModel(
                kind="llm_rerank",
                name="rk",
                api_key="db-key",
                api_key_hash="h1",
                priority=5,
                endpoint="https://db.test/v1",
                model="db-model",
            )
        )
        await session.commit()

    captured: dict[str, str] = {}

    class _FakeDelegate:
        async def chat(self, messages, model=None, **kwargs):
            captured["model"] = model
            return "ok"

    client = CredentialConfiguredLLMClient(
        "llm_rerank",
        sessions,
        LLMSettings(api_key="env-key"),
        fallback_model="env-model",
    )
    client._build_delegate = lambda config: _FakeDelegate()

    # 凭证 model 存在 → 覆盖调用方传入的 model
    await client.chat([{"role": "user", "content": "x"}], model="call-model")
    assert captured["model"] == "db-model"
    await engine.dispose()
