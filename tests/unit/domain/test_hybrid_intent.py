"""Hybrid adapter tests; all provider calls are local fakes."""

import asyncio

from oce.domain.services.llm.intent import (
    HybridIntentClassifier,
    LayaSoftSignalProvider,
    QueryIntent,
)


class _FakeLLM:
    def __init__(self, response: str, *, delay: float = 0.0):
        self.response = response
        self.delay = delay
        self.calls = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.response


async def test_hard_symbol_short_circuits_provider():
    llm = _FakeLLM('{"asks_api_usage":{"value":true,"confidence":1}}')
    provider = LayaSoftSignalProvider(llm_client=llm, model="local", timeout_ms=100)
    classifier = HybridIntentClassifier(model="local", soft_provider=provider)

    intent, decision = await classifier.classify_with_decision(
        "Where is `build_chunker` defined?"
    )

    assert intent is QueryIntent.SYMBOL
    assert decision.source == "hard"
    assert llm.calls == 0


async def test_soft_provider_is_used_only_for_ambiguous_semantics():
    llm = _FakeLLM(
        '{"asks_call_chain":{"value":false,"confidence":0.9},'
        '"asks_api_usage":{"value":false,"confidence":0.9},'
        '"asks_overview":{"value":true,"confidence":0.9},'
        '"asks_compound":{"value":false,"confidence":0.9},'
        '"asks_implementation":{"value":false,"confidence":0.9}}'
    )
    provider = LayaSoftSignalProvider(llm_client=llm, model="local", timeout_ms=100)
    classifier = HybridIntentClassifier(model="local", soft_provider=provider)

    intent, decision = await classifier.classify_with_decision("请说明这个模块的职责")

    assert intent is QueryIntent.OVERVIEW
    assert decision.source == "hybrid"
    assert decision.used_laya is True
    assert llm.calls == 1

    # Same normalized query is served from the bounded local cache.
    again, _ = await classifier.classify_with_decision("  请说明这个模块的职责  ")
    assert again is QueryIntent.OVERVIEW
    assert llm.calls == 1


async def test_ambiguous_symbol_usage_reaches_soft_provider():
    llm = _FakeLLM(
        '{"asks_call_chain":{"value":true,"confidence":0.95},'
        '"asks_api_usage":{"value":false,"confidence":0.95}}'
    )
    provider = LayaSoftSignalProvider(llm_client=llm, model="local", timeout_ms=100)
    classifier = HybridIntentClassifier(model="local", soft_provider=provider)

    intent, decision = await classifier.classify_with_decision(
        "How is `handler` used across the application?"
    )

    assert intent is QueryIntent.CALL_CHAIN
    assert decision.source == "hybrid"
    assert llm.calls == 1

async def test_malformed_or_timeout_provider_falls_back_without_error():
    malformed = _FakeLLM("not-json")
    classifier = HybridIntentClassifier(
        model="local",
        soft_provider=LayaSoftSignalProvider(
            llm_client=malformed, model="local", timeout_ms=100
        ),
    )
    intent, decision = await classifier.classify_with_decision("请说明这个模块的职责")
    assert intent is QueryIntent.FEATURE
    assert decision.source == "fallback"
    assert decision.fallback_reason == "soft_provider_error"

    slow = _FakeLLM('{"asks_overview":true}', delay=0.05)
    classifier = HybridIntentClassifier(
        model="local",
        soft_provider=LayaSoftSignalProvider(
            llm_client=slow, model="local", timeout_ms=1
        ),
    )
    intent, decision = await classifier.classify_with_decision("请说明这个模块的职责")
    assert intent is QueryIntent.FEATURE
    assert decision.source == "fallback"
