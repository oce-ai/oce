"""TypeSafe 判定源的契约与降级测试。

所有测试都注入假的 HTTP transport，绝不发起真实网络请求。
覆盖验收项 17 列出的每一种失败：网络异常、超时、401、429、529、
响应体非 JSON、缺少 choice 字段、置信度低于阈值。
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from oce.domain.services.intent.port import IntentPrediction, IntentProviderError
from oce.domain.services.intent.resolver import (
    IntentResolver,
    SOURCE_FALLBACK,
    SOURCE_PROVIDER,
    SOURCE_RULE_HARD,
    SOURCE_RULE_ONLY,
)
from oce.domain.services.intent.taxonomy import QueryIntent
from oce.infrastructure.intent.typesafe_provider import (
    TypeSafeIntentProvider,
    build_typesafe_provider,
)
from oce.infrastructure.intent.typesafe_provider import SYSTEM_ONE_PATH

# 判定表给出软结论、因而会咨询判定源的查询。
SOFT_QUERY = "Explain the retry behavior"
# 判定表给出硬结论、因而不咨询判定源的查询。
HARD_QUERY = "`foo_bar` 在哪里定义？"

API_KEY = "test-key-not-a-real-credential"


def _provider(handler, **kwargs) -> TypeSafeIntentProvider:
    """构造带假 transport 的 provider。"""
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    params = {
        "api_key": API_KEY,
        "base_url": "https://api.typesafe.test",
        "model": "jev-latest",
        "timeout_seconds": 2.0,
        "client": client,
    }
    params.update(kwargs)
    return TypeSafeIntentProvider(**params)


def _choice_response(choice: str, confidence: float, *, probabilities=None):
    body = {
        "model": "jev-1.13.0",
        "answers": {
            "intent": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities or {choice: confidence},
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    return httpx.Response(200, json=body)


# ── 契约：请求形状 ──────────────────────────────────────────────────────────


async def test_request_follows_system_one_contract() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return _choice_response("O", 0.9)

    provider = _provider(handler)
    prediction = await provider.predict("架构是怎样的")

    assert seen["url"].endswith(SYSTEM_ONE_PATH)
    assert seen["auth"] == f"Bearer {API_KEY}"
    body = seen["body"]
    assert body["state"] == "架构是怎样的"
    assert body["model"] == "jev-latest"
    question = body["questions"]["intent"]
    assert question["type"] == "choice"
    assert "instructions" in question
    # criteria 必须下发全部 8 个标签
    assert set(question["criteria"]) == {i.value for i in QueryIntent}

    assert prediction.intent is QueryIntent.OVERVIEW
    assert prediction.confidence == pytest.approx(0.9)
    assert prediction.model == "jev-1.13.0"
    assert prediction.latency_ms is not None


async def test_probabilities_and_confidence_are_read() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _choice_response("U", 0.83, probabilities={"U": 0.83, "S": 0.17})

    prediction = await _provider(handler).predict("哪些地方用到它")
    assert prediction.intent is QueryIntent.USAGE
    assert prediction.probabilities == {"U": 0.83, "S": 0.17}


async def test_cache_avoids_a_second_request() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _choice_response("F", 0.9)

    provider = _provider(handler)
    await provider.predict("retry behavior")
    await provider.predict("  retry   behavior  ")  # 规范化后同一条
    assert calls["n"] == 1
    assert provider.cache_size_used == 1


async def test_owned_http_client_drains_requests_before_close(monkeypatch) -> None:
    started = asyncio.Event()
    gate = asyncio.Event()

    async def handler(request):
        started.set()
        await gate.wait()
        return _choice_response("F", 0.9)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client)
    provider = TypeSafeIntentProvider(
        api_key=API_KEY, base_url="https://api.typesafe.test", model="jev-latest", timeout_seconds=2,
    )
    prediction = asyncio.create_task(provider.predict("retry behavior"))
    await started.wait()
    closing = asyncio.create_task(provider.aclose())
    await asyncio.sleep(0)
    assert not closing.done()
    assert not client.is_closed
    gate.set()
    assert (await prediction).intent == QueryIntent.FEATURE
    await closing
    assert client.is_closed
    with pytest.raises(IntentProviderError, match="closed"):
        await provider.predict("another query")


async def test_usage_is_reported_once_per_network_request_and_does_not_block_prediction() -> None:
    collected = []
    usage_started = asyncio.Event()
    usage_gate = asyncio.Event()

    async def on_usage(*args):
        usage_started.set()
        await usage_gate.wait()
        collected.append(args)

    provider = _provider(lambda _: _choice_response("F", 0.9), on_usage=on_usage, credential_id=42)
    prediction = await provider.predict("retry behavior")
    await usage_started.wait()
    assert prediction.intent == QueryIntent.FEATURE
    assert collected == []
    await provider.predict("retry behavior")
    usage_gate.set()
    await asyncio.gather(*provider._usage_tasks)
    assert collected == [(42, "llm", "jev-1.13.0", 100, 10)]


async def test_missing_usage_is_skipped_and_callback_errors_do_not_change_prediction() -> None:
    calls = []

    async def on_usage(*args):
        calls.append(args)
        raise RuntimeError("metrics unavailable")

    provider = _provider(lambda _: _choice_response("F", 0.9), on_usage=on_usage)
    assert (await provider.predict("query")).intent == QueryIntent.FEATURE
    await asyncio.gather(*provider._usage_tasks)
    assert len(calls) == 1

    provider = _provider(lambda _: httpx.Response(200, json={
        "answers": {"intent": {"choice": "F", "confidence": 0.9}}
    }), on_usage=on_usage)
    await provider.predict("other query")
    assert provider._usage_tasks == set()
    assert len(calls) == 1


# ── 无 key 时不发请求 ───────────────────────────────────────────────────────


def test_missing_api_key_yields_no_provider() -> None:
    built = build_typesafe_provider(
        api_key="",
        base_url="https://api.typesafe.test",
        model="jev-latest",
        timeout_seconds=2.0,
        cache_size=16,
    )
    assert built is None


def test_provider_refuses_construction_without_key() -> None:
    with pytest.raises(ValueError):
        TypeSafeIntentProvider(
            api_key="  ",
            base_url="https://api.typesafe.test",
            model="jev-latest",
            timeout_seconds=2.0,
        )


async def test_resolver_without_provider_is_rules_only() -> None:
    decision = await IntentResolver(provider=None).resolve(SOFT_QUERY)
    assert decision.intent is QueryIntent.FEATURE
    assert decision.source == SOURCE_RULE_ONLY
    assert decision.provider_intent is None
    assert decision.fallback_reason is None


# ── 失败模式：适配器把每种失败转成 IntentProviderError ──────────────────────


@pytest.mark.parametrize(
    "status,expected_reason",
    [
        (401, "http_401"),
        (422, "http_422"),
        (429, "http_429"),
        (529, "http_529"),
        (500, "http_500"),
    ],
)
async def test_http_error_statuses_become_provider_errors(
    status: int, expected_reason: str
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"error": "nope"})

    with pytest.raises(IntentProviderError) as excinfo:
        await _provider(handler).predict("q")
    assert excinfo.value.reason == expected_reason


async def test_timeout_becomes_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    with pytest.raises(IntentProviderError) as excinfo:
        await _provider(handler).predict("q")
    assert excinfo.value.reason == "timeout"


async def test_network_error_becomes_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    with pytest.raises(IntentProviderError) as excinfo:
        await _provider(handler).predict("q")
    assert excinfo.value.reason.startswith("transport_")


async def test_non_json_body_becomes_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    with pytest.raises(IntentProviderError) as excinfo:
        await _provider(handler).predict("q")
    assert excinfo.value.reason == "invalid_json"


async def test_missing_choice_field_becomes_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "jev", "answers": {"intent": {}}})

    with pytest.raises(IntentProviderError) as excinfo:
        await _provider(handler).predict("q")
    assert excinfo.value.reason == "unknown_choice"


async def test_missing_answer_becomes_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"model": "jev", "answers": {}})

    with pytest.raises(IntentProviderError) as excinfo:
        await _provider(handler).predict("q")
    assert excinfo.value.reason == "missing_answer"


async def test_unknown_label_becomes_provider_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _choice_response("ZZZ", 0.99)

    with pytest.raises(IntentProviderError) as excinfo:
        await _provider(handler).predict("q")
    assert excinfo.value.reason == "unknown_choice"


# ── 降级：分类器始终返回合法意图，不抛异常 ──────────────────────────────────


class _FailingProvider:
    def __init__(self, reason: str = "boom") -> None:
        self.reason = reason
        self.calls = 0

    async def predict(self, query: str) -> IntentPrediction:
        self.calls += 1
        raise IntentProviderError(self.reason)


class _ExplodingProvider:
    """抛未包装的异常，模拟适配器自身有 bug 的情况。"""

    async def predict(self, query: str) -> IntentPrediction:
        raise RuntimeError("unwrapped")


class _FixedProvider:
    def __init__(self, intent: QueryIntent, confidence: float) -> None:
        self.intent = intent
        self.confidence = confidence
        self.calls = 0

    async def predict(self, query: str) -> IntentPrediction:
        self.calls += 1
        return IntentPrediction(
            intent=self.intent,
            confidence=self.confidence,
            probabilities={self.intent.value: self.confidence},
            model="fake",
            latency_ms=1.0,
        )


@pytest.mark.parametrize(
    "reason",
    ["timeout", "http_401", "http_429", "http_529", "invalid_json", "missing_answer"],
)
async def test_provider_failures_fall_back_observably(reason: str) -> None:
    provider = _FailingProvider(reason)
    decision = await IntentResolver(provider=provider).resolve(SOFT_QUERY)

    assert decision.intent in set(QueryIntent)
    assert decision.source == SOURCE_FALLBACK
    assert decision.fallback_reason == reason
    assert decision.rule_intent is QueryIntent.FEATURE
    assert provider.calls == 1


async def test_unwrapped_exception_also_falls_back() -> None:
    decision = await IntentResolver(provider=_ExplodingProvider()).resolve(SOFT_QUERY)
    assert decision.source == SOURCE_FALLBACK
    assert decision.fallback_reason == "unexpected_RuntimeError"
    assert decision.intent in set(QueryIntent)


async def test_low_confidence_falls_back_but_records_the_prediction() -> None:
    provider = _FixedProvider(QueryIntent.OVERVIEW, confidence=0.20)
    decision = await IntentResolver(provider=provider, min_confidence=0.60).resolve(SOFT_QUERY)

    assert decision.source == SOURCE_FALLBACK
    assert decision.fallback_reason == "low_confidence"
    # 采纳的是判定表结论，但判定源的结论仍被记录下来供审计
    assert decision.intent is QueryIntent.FEATURE
    assert decision.provider_intent is QueryIntent.OVERVIEW
    assert decision.provider_confidence == pytest.approx(0.20)


# ── 判定来源可区分 ──────────────────────────────────────────────────────────


async def test_hard_rule_never_consults_the_provider() -> None:
    provider = _FixedProvider(QueryIntent.FEATURE, confidence=0.99)
    decision = await IntentResolver(provider=provider).resolve(HARD_QUERY)

    assert decision.source == SOURCE_RULE_HARD
    assert decision.intent is QueryIntent.SYMBOL
    assert provider.calls == 0, "硬事实不应产生付费调用"


async def test_provider_override_is_recorded() -> None:
    provider = _FixedProvider(QueryIntent.OVERVIEW, confidence=0.95)
    decision = await IntentResolver(provider=provider).resolve(SOFT_QUERY)

    assert decision.source == SOURCE_PROVIDER
    assert decision.intent is QueryIntent.OVERVIEW
    assert decision.rule_intent is QueryIntent.FEATURE
    assert "provider_override" in decision.reason


async def test_provider_agreement_is_recorded() -> None:
    provider = _FixedProvider(QueryIntent.FEATURE, confidence=0.95)
    decision = await IntentResolver(provider=provider).resolve(SOFT_QUERY)

    assert decision.source == SOURCE_PROVIDER
    assert decision.intent is QueryIntent.FEATURE
    assert "provider_agree" in decision.reason


async def test_decision_snapshot_is_serializable() -> None:
    provider = _FixedProvider(QueryIntent.OVERVIEW, confidence=0.95)
    decision = await IntentResolver(provider=provider).resolve(SOFT_QUERY)
    snapshot = decision.as_dict()
    json.dumps(snapshot)
    assert snapshot["source"] == SOURCE_PROVIDER
    assert snapshot["provider_intent"] == "O"
    assert snapshot["rule_intent"] == "F"
    assert "signals" in snapshot


def test_domain_layer_does_not_import_http() -> None:
    """域层不得 import 任何 HTTP 客户端或 SDK。"""
    import pathlib

    domain_dir = pathlib.Path(__file__).resolve().parents[3] / "src" / "oce" / "domain"
    offenders = []
    for path in domain_dir.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("import httpx", "import requests", "from httpx", "typesafe_sdk", "import openai"):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, f"域层出现出站依赖: {offenders}"
