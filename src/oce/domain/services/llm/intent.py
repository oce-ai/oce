"""查询意图分类器 - LLM-based 7-category classifier

意图分类：
- S (SYMBOL): 符号定义查询
- C (CALL_CHAIN): 调用链/流程查询  
- R (REFERENCE): 引用/使用位置查询
- P (PATH): 文件路径查询
- F (FEATURE): 功能实现查询
- O (OVERVIEW): 架构/机制概览查询
- M (COMPOUND): 多条件复合查询
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import OrderedDict
from dataclasses import replace
from enum import Enum
from hashlib import sha256
from time import perf_counter
from typing import Any, Mapping

from oce.domain.services.llm.client import LLMClient
from oce.domain.services.llm.prompts import (
    INTENT_SOFT_PROMPT_VERSION,
    INTENT_SOFT_SYSTEM_PROMPT,
    INTENT_SOFT_USER_TEMPLATE,
    INTENT_SYSTEM_PROMPT,
    INTENT_USER_TEMPLATE,
)
from oce.domain.services.intent_resolver import (
    IntentDecision,
    IntentResolver,
    SoftSignals,
)
from oce.domain.services.query_classifier import (
    QueryIntent as HardQueryIntent,
    extract_hard_signals,
)


class QueryIntent(str, Enum):
    """查询意图枚举"""
    
    SYMBOL = "S"          # 符号定义
    CALL_CHAIN = "C"      # 调用链
    REFERENCE = "R"       # 引用位置
    PATH = "P"            # 文件路径
    FEATURE = "F"         # 功能实现
    OVERVIEW = "O"        # 架构概览
    COMPOUND = "M"        # 复合查询


# Label 映射
LABEL_TO_INTENT = {
    'S': QueryIntent.SYMBOL,
    'C': QueryIntent.CALL_CHAIN,
    'R': QueryIntent.REFERENCE,
    'P': QueryIntent.PATH,
    'F': QueryIntent.FEATURE,
    'O': QueryIntent.OVERVIEW,
    'M': QueryIntent.COMPOUND,
}


class IntentClassifier:
    """查询意图分类器（LLM-based）"""
    
    def __init__(self, llm_client: LLMClient, model: str):
        """
        Args:
            llm_client: LLM 客户端（需支持 chat 方法）
            model: 模型名称
        """
        self.llm_client = llm_client
        self.model = model
    
    async def classify(self, query: str) -> QueryIntent:
        """分类查询意图
        
        Args:
            query: 查询文本
            
        Returns:
            QueryIntent 枚举值
        """
        user_prompt = INTENT_USER_TEMPLATE.format(query=query)
        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        response = await self.llm_client.chat(
            messages=messages,
            model=self.model,
            temperature=0,
            max_tokens=2,
        )
        
        label = response.strip().upper()
        return LABEL_TO_INTENT.get(label, QueryIntent.FEATURE)


SOFT_SIGNAL_NAMES = (
    "asks_call_chain",
    "asks_api_usage",
    "asks_overview",
    "asks_compound",
    "asks_implementation",
)


class SoftSignalProviderError(RuntimeError):
    """Provider transport or response error; callers must fall back safely."""


class LayaSoftSignalProvider:
    """Adapt one JSON chat completion into provider-neutral ``SoftSignals``.

    The adapter intentionally has no knowledge of the final seven-way taxonomy.
    It is safe to point at a local Laya-compatible endpoint or any configured
    OpenAI-compatible credential; no endpoint or key is embedded here.
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        model: str,
        timeout_ms: int = 50,
        min_confidence: float = 0.60,
        cache_size: int = 1024,
        provider_name: str = "laya",
        prompt_version: str = INTENT_SOFT_PROMPT_VERSION,
    ) -> None:
        self.llm_client = llm_client
        self.model = model
        self.timeout_ms = max(1, int(timeout_ms))
        self.min_confidence = min_confidence
        self.cache_size = max(0, int(cache_size))
        self.provider_name = provider_name
        self.prompt_version = prompt_version
        self._cache: OrderedDict[str, SoftSignals] = OrderedDict()

    @property
    def cache_size_used(self) -> int:
        return len(self._cache)

    def _cache_key(self, query: str) -> str:
        normalized = " ".join((query or "").split()).casefold()
        material = f"{normalized}\n{self.prompt_version}\n{self.model}"
        return sha256(material.encode("utf-8")).hexdigest()

    async def classify(self, query: str, *, hard_signals: Any | None = None) -> SoftSignals:
        key = self._cache_key(query)
        if self.cache_size:
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                return cached

        hard = hard_signals or extract_hard_signals(query)
        hard_summary = {
            "has_concrete_symbol": hard.has_concrete_symbol,
            "identifier_count": hard.identifier_count,
            "has_filename": hard.has_filename,
            "has_explicit_path": hard.has_explicit_path,
            "has_flow_delimiter": hard.has_flow_delimiter,
            "has_independent_clauses": hard.has_independent_clauses,
        }
        messages = [
            {"role": "system", "content": INTENT_SOFT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": INTENT_SOFT_USER_TEMPLATE.format(
                    query=query,
                    hard_signals=json.dumps(hard_summary, ensure_ascii=False),
                ),
            },
        ]
        started = perf_counter()
        try:
            response = await asyncio.wait_for(
                self.llm_client.chat(
                    messages=messages,
                    model=self.model,
                    temperature=0,
                    max_tokens=400,
                ),
                timeout=self.timeout_ms / 1000,
            )
            payload = _parse_json_object(response)
            result = _soft_signals_from_payload(
                payload,
                provider=self.provider_name,
                model=self.model,
                latency_ms=(perf_counter() - started) * 1000,
            )
        except Exception as exc:
            raise SoftSignalProviderError(type(exc).__name__) from exc

        if self.cache_size:
            self._cache[key] = result
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return result


def _parse_json_object(response: str) -> Mapping[str, Any]:
    """Parse strict JSON, fenced JSON, or a JSON object embedded in prose."""
    text = (response or "").strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].lstrip()
    decoder = json.JSONDecoder()
    candidates = [text]
    start = text.find("{")
    if start >= 0:
        candidates.append(text[start:])
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, Mapping):
            return value
    raise ValueError("soft signal response is not a JSON object")


def _soft_signals_from_payload(
    payload: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    latency_ms: float,
) -> SoftSignals:
    """Normalize vendor output; malformed individual fields become unknown."""
    # Accept the common Laya ``answers.intent`` envelope as well as the
    # provider-neutral top-level object documented by OCE.
    nested = payload.get("answers") if isinstance(payload, Mapping) else None
    if isinstance(nested, Mapping) and isinstance(nested.get("intent"), Mapping):
        payload = nested["intent"]
    normalized: dict[str, Any] = {
        name: payload.get(name) for name in SOFT_SIGNAL_NAMES
    }
    normalized["provider"] = provider
    normalized["model"] = model
    normalized["latency_ms"] = latency_ms
    normalized["used_provider"] = True
    return SoftSignals.from_mapping(normalized, provider=provider, model=model, latency_ms=latency_ms)


_HARD_TO_LLM = {
    HardQueryIntent.SYMBOL: QueryIntent.SYMBOL,
    HardQueryIntent.CALL_CHAIN: QueryIntent.CALL_CHAIN,
    HardQueryIntent.REFERENCE: QueryIntent.REFERENCE,
    HardQueryIntent.PATH: QueryIntent.PATH,
    HardQueryIntent.FEATURE: QueryIntent.FEATURE,
    HardQueryIntent.OVERVIEW: QueryIntent.OVERVIEW,
    HardQueryIntent.COMPOUND: QueryIntent.COMPOUND,
}


class HybridIntentClassifier:
    """Hard-signal extractor + optional soft provider + deterministic resolver."""

    def __init__(
        self,
        *,
        model: str,
        soft_provider: LayaSoftSignalProvider | None = None,
        min_confidence: float = 0.60,
        resolver_version: str = "hybrid-v2",
        legacy_static_reference_as_symbol: bool = False,
    ) -> None:
        self.model = model
        self.soft_provider = soft_provider
        self.resolver = IntentResolver(
            min_confidence=min_confidence,
            resolver_version=resolver_version,
            legacy_static_reference_as_symbol=legacy_static_reference_as_symbol,
        )
        self.resolver_version = resolver_version
        self.legacy_static_reference_as_symbol = legacy_static_reference_as_symbol
        self.last_decision: IntentDecision | None = None

    def _hard_short_circuit(self, query: str, hard: Any) -> bool:
        """Avoid a provider call whenever the structure already fixes the branch."""
        if hard.is_empty:
            return True
        if hard.has_concrete_symbol:
            if hard.has_flow_delimiter or hard.has_api_usage_marker or hard.has_definition_marker:
                return True
            if hard.has_reference_marker:
                # Static reference lookup is a hard branch in hybrid-v2 (R);
                # generic ``used/how is it used`` wording is left to soft
                # signals so C versus R can be resolved semantically.
                static_reference = re.search(
                    r"(?:哪些文件|哪些地方|引用位置|被引用|"
                    r"in\s+which\s+files?|static\s+references?|where\s+is\s+.+\s+used)",
                    query,
                    re.IGNORECASE,
                )
                return static_reference is not None
            return True
        if hard.has_independent_clauses or hard.has_explicit_path:
            return True
        return bool(hard.has_overview_marker or hard.has_implementation_marker)

    async def classify_with_decision(self, query: str) -> tuple[QueryIntent, IntentDecision]:
        hard = extract_hard_signals(query)
        soft = SoftSignals()
        fallback_reason: str | None = None
        if self.soft_provider is not None and not self._hard_short_circuit(query, hard):
            try:
                soft = await self.soft_provider.classify(query, hard_signals=hard)
            except Exception:
                # A provider is an optional semantic hint.  Any transport,
                # timeout, credential, or adapter failure becomes unknown soft
                # signals; it must never break the retrieval request.
                fallback_reason = "soft_provider_error"
        decision = self.resolver.resolve(query, hard=hard, soft=soft)
        if decision.resolver_version != self.resolver_version:
            decision = replace(decision, resolver_version=self.resolver_version)
        if fallback_reason:
            decision = replace(decision, source="fallback", fallback_reason=fallback_reason)
        self.last_decision = decision
        return _HARD_TO_LLM[decision.intent], decision

    async def classify(self, query: str) -> QueryIntent:
        intent, _ = await self.classify_with_decision(query)
        return intent


class HeuristicIntentClassifier:
    """Compatibility classifier for the explicit ``heuristic`` mode."""

    async def classify(self, query: str) -> QueryIntent:
        from oce.domain.services.query_classifier import classify_query_intent

        return _HARD_TO_LLM[classify_query_intent(query)]


class ShadowIntentClassifier:
    """Run hybrid classification for observation while returning heuristic output."""

    def __init__(self, hybrid: HybridIntentClassifier) -> None:
        self.hybrid = hybrid
        self.heuristic = HeuristicIntentClassifier()
        self.last_decision: IntentDecision | None = None

    async def classify_with_decision(self, query: str) -> tuple[QueryIntent, IntentDecision]:
        _, shadow_decision = await self.hybrid.classify_with_decision(query)
        main_intent = await self.heuristic.classify(query)
        inverse = {value: key for key, value in _HARD_TO_LLM.items()}
        decision = replace(
            shadow_decision,
            intent=inverse[main_intent],
            source="heuristic",
            reason="shadow_mainline_heuristic",
        )
        self.last_decision = decision
        return main_intent, decision

    async def classify(self, query: str) -> QueryIntent:
        intent, _ = await self.classify_with_decision(query)
        return intent
