"""Deterministic resolver for the hybrid intent classifier.

The flat seven-way LLM classifier is intentionally not used here.  A query's
structural facts (symbols, paths and independent clauses) are extracted first;
an optional provider may add a handful of tri-state semantic signals.  This
module is pure domain logic: it has no network, configuration, database or
model dependency and is therefore safe to use as the final fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from oce.domain.services.query_classifier import (
    HardSignals,
    QueryIntent,
    extract_hard_signals,
)


@dataclass(frozen=True, slots=True)
class SoftSignal:
    """Convenience representation for one tri-state signal plus confidence."""

    value: bool | None
    confidence: float | None = None
_API_USAGE_RE = re.compile(
    r"(?:怎么调用|如何调用|怎样调用|怎么使用|如何使用|调用示例|使用示例|传参|参数|签名|"
    r"返回值|返回类型|\bhow\s+(?:do\s+i|to)\s+(?:call|use)\b|\barguments?\b|"
    r"\bparameters?\b|\bsignature\b|\busage\s+example\b|\binvocation\b|"
    r"example\s+call|correct\s+way\s+to\s+call|callers?\s+(?:invoke|call|use)|"
    r"怎么用|如何用|怎样用|API\s+怎么用)",
    re.IGNORECASE,
)
_FLOW_RE = re.compile(
    r"(?:调用链|调用路径|完整流程|执行路径|触发路径|触发后.{0,40}(?:执行|调用|依次)|"
    r"依次调用|一路执行|调用哪些|端到端|全链路|从.{0,40}到|入口.{0,40}到|"
    r"\bcall\s+chain\b|\bcall\s+path\b|\bexecution\s+path\b|\btrigger(?:ed)?\b|"
    r"\bgets?\s+(?:invoked|called)\b.{0,60}\breaches?\b|"
    r"\binvoked\b.{0,60}\breaches?\b|\btriggered\b.{0,60}\bcalls?\b|"
    r"\bend[- ]to[- ]end\b|\bentry\s+points?\b|\btrace\b|\bworkflow\b|\bpipeline\b|"
    r"\bwhat\s+does\s+.+\s+call\s+next\b)",
    re.IGNORECASE,
)
_OVERVIEW_RE = re.compile(
    r"(?:架构|机制|状态管理|状态流转|调度|事件处理|跨层|各层|数据流|系统级|整体|如何协作|"
    r"\barchitecture\b|\bmechanism\b|\bstate\s+(?:flow|management)\b|\bscheduling\b|"
    r"\bevent\s+handling\b|\bcross[- ]layer\b|\bdata\s+flow\b|\bsystem[- ]level\b|"
    r"\boverall\s+(?:architecture|design|flow|interaction)\b)",
    re.IGNORECASE,
)
_IMPLEMENTATION_RE = re.compile(
    r"(?:功能|业务规则|行为|实现逻辑|处理逻辑|功能实现|哪里实现|实现代码|"
    r"\bfeature\b|\bbehavior\b|\bbusiness\s+rule\b|\bimplemented\b|"
    r"\bimplementation\b|\bhandle(?:d|s)?\b|\blogic\b|\bwhere\s+implemented\b)",
    re.IGNORECASE,
)
_COMPOUND_CONNECTOR_RE = re.compile(
    r"(?:以及|并且|同时|另外|分别|并说明|并解释|并分析|和|与|及|\band\b|\balso\b|\bas\s+well\s+as\b|\bplus\b)",
    re.IGNORECASE,
)
_TASK_RE = re.compile(
    r"(?:定义|源码|实现|调用|使用|说明|解释|查找|找出|定位|处理|逻辑|行为|"
    r"流程|机制|功能|恢复|删除|注册|分析|"
    r"where|how|what|which|show|find|explain|describe|locate|call|use|handle|"
    r"implemented|behavior|implementation|feature|rollout|recovery|logic|handling)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SoftSignals:
    """Optional semantic signals.  ``None`` means unknown, not false."""

    asks_call_chain: bool | SoftSignal | None = None
    asks_api_usage: bool | SoftSignal | None = None
    asks_overview: bool | SoftSignal | None = None
    asks_compound: bool | SoftSignal | None = None
    asks_implementation: bool | SoftSignal | None = None
    confidence: Mapping[str, float] = field(default_factory=dict)
    provider: str | None = None
    model: str | None = None
    latency_ms: float | None = None
    used_provider: bool = False

    def trusted(self, name: str, minimum: float = 0.60) -> bool | None:
        """Return a signal only when its optional confidence clears a threshold."""
        value = getattr(self, name)
        if isinstance(value, SoftSignal):
            if value.value is None:
                return None
            if value.confidence is not None and value.confidence < minimum:
                return None
            return bool(value.value)
        if value is None:
            return None
        confidence = self.confidence.get(name)
        if confidence is not None and confidence < minimum:
            return None
        return bool(value)

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any] | None,
        *,
        provider: str | None = None,
        model: str | None = None,
        latency_ms: float | None = None,
    ) -> "SoftSignals":
        """Normalize adapter output without letting malformed fields decide."""
        value = value or {}
        names = (
            "asks_call_chain",
            "asks_api_usage",
            "asks_overview",
            "asks_compound",
            "asks_implementation",
        )
        parsed: dict[str, bool | None] = {}
        confidence: dict[str, float] = {}
        for name in names:
            raw = value.get(name)
            conf: float | None = None
            if isinstance(raw, Mapping):
                raw_value = raw.get("value")
                try:
                    conf = float(raw.get("confidence"))
                except (TypeError, ValueError):
                    conf = None
            else:
                raw_value = raw
            if isinstance(raw_value, str):
                normalized_value = raw_value.strip().casefold()
                if normalized_value in {"true", "yes", "1"}:
                    raw_value = True
                elif normalized_value in {"false", "no", "0"}:
                    raw_value = False
            if isinstance(raw_value, bool):
                parsed[name] = raw_value
            else:
                parsed[name] = None
            if conf is not None and 0.0 <= conf <= 1.0:
                confidence[name] = conf
        used = value.get("used_provider", provider or value.get("provider"))
        if isinstance(used, str):
            used = used.strip().casefold() in {"true", "yes", "1"}
        return cls(
            **parsed,
            confidence=confidence,
            provider=provider or value.get("provider"),
            model=model or value.get("model"),
            latency_ms=latency_ms,
            used_provider=bool(used),
        )


@dataclass(frozen=True, slots=True)
class IntentDecision:
    """Auditable final decision returned by :func:`resolve_intent`."""

    intent: QueryIntent
    reason: str
    source: str  # hard, hybrid, heuristic, fallback
    hard_signals: HardSignals
    soft_signals: SoftSignals = field(default_factory=SoftSignals)
    fallback_reason: str | None = None
    # ``hybrid-v2`` deliberately separates static-reference semantics from the
    # frozen legacy 7-way benchmark.  Keep the version in the decision so an
    # audit record (and any provider cache keyed by it) cannot be mistaken for
    # the old benchmark-compatible resolver.
    resolver_version: str = "hybrid-v2"

    @property
    def used_provider(self) -> bool:
        return self.soft_signals.used_provider

    @property
    def used_laya(self) -> bool:
        """Compatibility alias used by audit/report code and the design document."""
        return self.used_provider

    def as_dict(self) -> dict[str, Any]:
        def serializable(value: Any) -> Any:
            if isinstance(value, SoftSignal):
                return {"value": value.value, "confidence": value.confidence}
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, tuple):
                return [serializable(item) for item in value]
            if isinstance(value, Mapping):
                return {str(key): serializable(item) for key, item in value.items()}
            return value

        return {
            "intent": self.intent.value,
            "reason": self.reason,
            "source": self.source,
            "fallback_reason": self.fallback_reason,
            "resolver_version": self.resolver_version,
            "hard_signals": {
                name: serializable(getattr(self.hard_signals, name))
                for name in self.hard_signals.__dataclass_fields__
            },
            "soft_signals": {
                name: serializable(getattr(self.soft_signals, name))
                for name in self.soft_signals.__dataclass_fields__
                if name != "confidence"
            },
        }


def _strong_compound(query: str, hard: HardSignals) -> bool:
    """Recognize two task-shaped clauses without treating every ``and`` as M."""
    if not hard.has_independent_clauses:
        return False
    parts = [p.strip() for p in _COMPOUND_CONNECTOR_RE.split(query) if p.strip()]
    if len(parts) < 2:
        return True
    return sum(bool(_TASK_RE.search(part)) for part in parts) >= 2


def _bool_signal(soft: SoftSignals, name: str, minimum: float) -> bool | None:
    return soft.trusted(name, minimum)


def resolve_intent(
    query: str,
    hard: HardSignals | None = None,
    soft: SoftSignals | None = None,
    *,
    min_confidence: float = 0.60,
    legacy_static_reference_as_symbol: bool = False,
) -> IntentDecision:
    """Resolve one query using hard facts first and optional soft signals second.

    Concrete symbols outrank filename/path cues.  A compound result requires
    structural evidence; a provider may confirm it but cannot invent it.

    ``hybrid-v2`` treats a concrete symbol's static reference lookup (for
    example, ``in which files is `foo` used``) as ``REFERENCE``.  The frozen
    84-case prompt benchmark historically called those same queries ``S``;
    callers that need to reproduce that *legacy* regression can opt into
    ``legacy_static_reference_as_symbol`` explicitly.  It must not be the
    production default because it would make the resolver depend on a test-set
    label convention rather than the documented intent boundary.
    """
    text = (query or "").strip()
    hard = hard or extract_hard_signals(text)
    soft = soft or SoftSignals()

    if hard.is_empty:
        return IntentDecision(QueryIntent.FEATURE, "empty_or_unclassified_query", "fallback", hard, soft)

    flow = hard.has_flow_delimiter or _FLOW_RE.search(text) is not None
    flow_soft = _bool_signal(soft, "asks_call_chain", min_confidence)
    api_soft = _bool_signal(soft, "asks_api_usage", min_confidence)
    overview = hard.has_overview_marker or _OVERVIEW_RE.search(text) is not None
    overview_soft = _bool_signal(soft, "asks_overview", min_confidence)
    implementation = hard.has_implementation_marker or _IMPLEMENTATION_RE.search(text) is not None
    implementation_soft = _bool_signal(soft, "asks_implementation", min_confidence)
    compound_soft = _bool_signal(soft, "asks_compound", min_confidence)

    # Independent retrieval goals are structural and outrank incidental
    # symbols/path cues inside either clause.  A provider may veto a false
    # compound signal, but it cannot make a structurally independent query
    # disappear.
    compound_hard = _strong_compound(text, hard)
    if compound_hard and (compound_soft is not False):
        return IntentDecision(QueryIntent.COMPOUND, "independent_clauses_confirmed", "hybrid" if soft.used_provider else "hard", hard, soft)

    # Concrete symbols outrank filename/path wording for single-goal queries.
    if hard.has_concrete_symbol:
        if flow or flow_soft is True:
            return IntentDecision(QueryIntent.CALL_CHAIN, "symbol_with_multi_step_flow", "hybrid" if soft.used_provider else "hard", hard, soft)
        if api_soft is True or _API_USAGE_RE.search(text) is not None:
            return IntentDecision(QueryIntent.REFERENCE, "symbol_single_point_usage", "hybrid" if soft.used_provider else "hard", hard, soft)
        # Hybrid-v2 distinguishes static reference lookup from a definition or
        # implementation lookup.  The old 84-case benchmark labelled the
        # former as S; retain that convention only behind an explicit legacy
        # switch (used by the frozen regression harness, never by production
        # defaults).
        if (
            hard.has_reference_marker
            and not hard.has_definition_marker
            and not legacy_static_reference_as_symbol
        ):
            return IntentDecision(QueryIntent.REFERENCE, "static_reference_scope", "hybrid" if soft.used_provider else "hard", hard, soft)
        return IntentDecision(QueryIntent.SYMBOL, "concrete_symbol_default", "hybrid" if soft.used_provider else "hard", hard, soft)

    # PATH is valid only when the request has a real file/path shape and is not
    # asking for implementation or system semantics.  Generic ``where`` alone
    # is intentionally insufficient.
    if (
        hard.has_explicit_path
        and not implementation
        and implementation_soft is not True
        and not overview
        and overview_soft is not True
    ):
        return IntentDecision(QueryIntent.PATH, "explicit_path_without_function_semantics", "hybrid" if soft.used_provider else "hard", hard, soft)

    if overview or overview_soft is True:
        return IntentDecision(QueryIntent.OVERVIEW, "system_level_semantics", "hybrid" if soft.used_provider else "heuristic", hard, soft)

    # If structural evidence is ambiguous, a confirmed semantic compound can
    # still select M; otherwise the conservative compatible default is FEATURE.
    if compound_soft is True and hard.has_independent_clauses:
        return IntentDecision(QueryIntent.COMPOUND, "provider_confirmed_independent_clauses", "hybrid", hard, soft)
    return IntentDecision(QueryIntent.FEATURE, "feature_or_conservative_default", "hybrid" if soft.used_provider else "fallback", hard, soft)


class IntentResolver:
    """Small callable facade useful for dependency injection and tests."""

    def __init__(
        self,
        *,
        min_confidence: float = 0.60,
        resolver_version: str = "hybrid-v2",
        legacy_static_reference_as_symbol: bool = False,
    ) -> None:
        self.min_confidence = min_confidence
        self.resolver_version = resolver_version
        self.legacy_static_reference_as_symbol = legacy_static_reference_as_symbol

    def resolve(
        self,
        query: str | HardSignals = "",
        soft: SoftSignals | None = None,
        *,
        hard: HardSignals | None = None,
    ) -> IntentDecision:
        # Accept ``resolve(hard, soft)`` as a convenience for pure-domain tests
        # while keeping ``resolve(query, hard=..., soft=...)`` for production.
        if isinstance(query, HardSignals):
            hard = query
            query = ""
        decision = resolve_intent(
            query,
            hard,
            soft,
            min_confidence=self.min_confidence,
            legacy_static_reference_as_symbol=self.legacy_static_reference_as_symbol,
        )
        if decision.resolver_version != self.resolver_version:
            # ``IntentDecision`` is immutable; replacing here keeps the pure
            # function's default useful while allowing composition roots to
            # pin a version in audit/cache metadata.
            decision = replace(decision, resolver_version=self.resolver_version)
        return decision
