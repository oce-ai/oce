"""Intent credential selection, transport compatibility, and connection retirement."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oce.domain.services.intent.port import IntentPrediction, IntentProviderError
from oce.domain.services.intent.resolver import IntentResolver, SOURCE_FALLBACK
from oce.domain.services.intent.taxonomy import QueryIntent
from oce.infrastructure.intent.credential_provider import CredentialConfiguredIntentProvider, IntentRuntimeConfig
from oce.infrastructure.intent.openai_provider import OpenAIIntentProvider
from oce.infrastructure.intent.typesafe_provider import TypeSafeIntentProvider
from oce.infrastructure.persistence.models import ModelCredentialModel
from oce.shared.config.settings import LLMSettings, RetrievalSettings


@pytest.fixture
async def sessions():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(ModelCredentialModel.__table__.create)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _provider(sessions, **retrieval):
    return CredentialConfiguredIntentProvider(
        sessions,
        RetrievalSettings(_env_file=None, intent_provider_api_key="", **retrieval),
        LLMSettings(_env_file=None, api_key=""),
    )


def _credential(name, **values):
    fields = dict(
        kind="intent", name=name, api_key=f"dummy-{name}", api_key_hash=name,
        endpoint="https://api.typesafe.ai/v1", model="jev-latest", priority=5,
        status="active", timeout_seconds=7,
    )
    fields.update(values)
    return ModelCredentialModel(**fields)


async def test_active_intent_credential_wins_by_priority_then_id(sessions):
    async with sessions() as session:
        first = _credential("first")
        session.add_all([
            _credential("disabled", priority=0, status="disabled"),
            _credential("other-kind", priority=0, kind="query_rewrite"),
            _credential("low-priority", priority=100),
            first,
            _credential("second"),
        ])
        await session.commit()
        first_id = first.id
    provider = _provider(sessions)
    config = await provider._resolve_config()
    assert config.credential_id == first_id
    assert config.api_key == "dummy-first"
    assert config.transport == "typesafe"
    assert config.timeout_seconds == 7
    delegate = provider._build_delegate(config)
    assert isinstance(delegate, TypeSafeIntentProvider)
    assert delegate.base_url == "https://api.typesafe.ai"
    await delegate.aclose()


async def test_database_overrides_dedicated_environment_key(sessions):
    async with sessions() as session:
        session.add(_credential("db"))
        await session.commit()
    provider = CredentialConfiguredIntentProvider(
        sessions, RetrievalSettings(_env_file=None, intent_provider_api_key="dummy-env"),
        LLMSettings(_env_file=None, api_key="dummy-legacy"),
    )
    assert (await provider._resolve_config()).api_key == "dummy-db"


async def test_dedicated_fallback_accepts_secretstr_and_own_model(sessions):
    provider = CredentialConfiguredIntentProvider(
        sessions,
        RetrievalSettings(_env_file=None, intent_provider_api_key="dummy-dedicated", intent_provider_model="jev-fixed", intent_provider_timeout_seconds=4),
        LLMSettings(_env_file=None, api_key="dummy-openai", model="legacy-model"),
    )
    config = await provider._resolve_config()
    assert config.transport == "typesafe"
    assert config.api_key == "dummy-dedicated"
    assert config.model == "jev-fixed"
    assert config.timeout_seconds == 4


async def test_legacy_llm_environment_uses_openai_transport(sessions):
    provider = CredentialConfiguredIntentProvider(
        sessions, RetrievalSettings(_env_file=None, intent_provider_api_key=""),
        LLMSettings(_env_file=None, api_key="dummy-openai", base_url="https://legacy.test/v1", model="legacy-model"),
    )
    config = await provider._resolve_config()
    delegate = provider._build_delegate(config)
    assert isinstance(delegate, OpenAIIntentProvider)
    assert delegate._client.base_url == "https://legacy.test/v1"
    assert delegate._client.api_key == "dummy-openai"
    assert delegate.model == "legacy-model"


@pytest.mark.parametrize("model", ["old-chat-model", "jev-latest"])
async def test_explicit_openai_credential_endpoint_is_authoritative(sessions, model):
    async with sessions() as session:
        session.add(_credential("legacy", endpoint="https://legacy.test/v1", model=model, provider="typesafe"))
        await session.commit()
    provider = _provider(sessions)
    config = await provider._resolve_config()
    assert config.transport == "openai"
    assert provider._build_delegate(config)._client.base_url == "https://legacy.test/v1"


async def test_no_credentials_uses_rule_fallback_and_does_not_repeat_database_lookup(sessions):
    provider = _provider(sessions)
    provider._resolve_config = AsyncMock(return_value=None)
    resolver = IntentResolver(provider=provider)
    first = await resolver.resolve("Explain the retry behavior")
    second = await resolver.resolve("Explain the retry behavior")
    assert first.source == SOURCE_FALLBACK
    assert first.fallback_reason == "no_credentials"
    assert first.intent == second.intent == QueryIntent.FEATURE
    assert provider._resolve_config.await_count == 1
    await provider.close()


class _Delegate:
    def __init__(self, intent=QueryIntent.FEATURE, *, gate=None):
        self.intent = intent
        self.gate = gate
        self.started = asyncio.Event()
        self.closed = False

    async def predict(self, query):
        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        assert not self.closed
        return IntentPrediction(intent=self.intent, confidence=0.9)

    async def aclose(self):
        self.closed = True


def _fake_config(model="old"):
    return IntentRuntimeConfig("typesafe", "dummy", "https://api.typesafe.ai", model, 3)


async def test_reload_retires_old_delegate_only_after_prediction_finishes(sessions):
    gate = asyncio.Event()
    old = _Delegate(gate=gate)
    new = _Delegate(QueryIntent.OVERVIEW)
    provider = _provider(sessions)
    provider._resolve_config = AsyncMock(side_effect=[_fake_config(), _fake_config("new")])
    provider._build_delegate = lambda config: old if config.model == "old" else new
    pending = asyncio.create_task(provider.predict("retry behavior"))
    await old.started.wait()
    assert await provider.reload() == 1
    assert not old.closed
    assert (await provider.predict("architecture")).intent == QueryIntent.OVERVIEW
    gate.set()
    assert (await pending).intent == QueryIntent.FEATURE
    assert old.closed
    await provider.close()
    assert new.closed


async def test_reload_picks_up_credential_changes_in_database(sessions):
    async with sessions() as session:
        credential = _credential("db")
        session.add(credential)
        await session.commit()
    provider = _provider(sessions)
    provider._build_delegate = lambda config: _Delegate(QueryIntent.OVERVIEW if config.model == "new" else QueryIntent.FEATURE)
    assert (await provider.predict("query")).intent == QueryIntent.FEATURE
    async with sessions() as session:
        credential = await session.get(ModelCredentialModel, credential.id)
        credential.model = "new"
        await session.commit()
    assert await provider.reload() == 1
    assert (await provider.predict("query")).intent == QueryIntent.OVERVIEW
    await provider.close()


async def test_close_waits_for_predict_before_first_database_await(sessions):
    lookup_started = asyncio.Event()
    lookup_gate = asyncio.Event()
    delegate = _Delegate()
    provider = _provider(sessions)

    async def config():
        lookup_started.set()
        await lookup_gate.wait()
        return _fake_config()

    provider._resolve_config = config
    provider._build_delegate = lambda _: delegate
    prediction = asyncio.create_task(provider.predict("query"))
    await lookup_started.wait()
    closing = asyncio.create_task(provider.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    lookup_gate.set()
    assert (await prediction).intent == QueryIntent.FEATURE
    await closing
    assert delegate.closed
    with pytest.raises(IntentProviderError, match="closed"):
        await provider.predict("query")


async def test_cancelled_predict_can_still_drain_and_close(sessions):
    delegate = _Delegate(gate=asyncio.Event())
    provider = _provider(sessions)
    provider._resolve_config = AsyncMock(return_value=_fake_config())
    provider._build_delegate = lambda _: delegate
    prediction = asyncio.create_task(provider.predict("query"))
    await delegate.started.wait()
    prediction.cancel()
    with pytest.raises(asyncio.CancelledError):
        await prediction
    await asyncio.wait_for(provider.close(), timeout=1)
    assert delegate.closed


async def test_openai_adapter_uses_all_labels_and_parses_json():
    client = AsyncMock()
    client.chat.return_value = '{"choice":"U","confidence":0.85}'
    provider = OpenAIIntentProvider(client, model="legacy-model", timeout_seconds=1)
    prediction = await provider.predict("Which callers use the helper?")
    assert prediction.intent == QueryIntent.USAGE
    system = client.chat.call_args.kwargs["messages"][0]["content"]
    assert all(f'"{intent.value}"' in system for intent in QueryIntent)
    assert client.chat.call_args.kwargs["model"] == "legacy-model"


async def test_openai_adapter_timeout_falls_back():
    client = AsyncMock()

    async def wait_forever(**kwargs):
        await asyncio.Event().wait()

    client.chat.side_effect = wait_forever
    provider = OpenAIIntentProvider(client, model="legacy-model", timeout_seconds=0.01)
    with pytest.raises(IntentProviderError, match="timeout"):
        await provider.predict("query")


@pytest.mark.parametrize("confidence", ["NaN", "Infinity", "-1", "2"])
async def test_openai_adapter_rejects_invalid_confidence(confidence):
    client = AsyncMock()
    client.chat.return_value = f'{{"choice":"F","confidence":{confidence}}}'
    with pytest.raises(IntentProviderError, match="invalid_confidence"):
        await OpenAIIntentProvider(client, model="model", timeout_seconds=1).predict("query")
