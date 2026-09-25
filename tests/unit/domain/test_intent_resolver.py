"""Pure hybrid intent resolver regression tests."""

import json

from oce.domain.services.intent_resolver import (
    IntentDecision,
    IntentResolver,
    SoftSignal,
    SoftSignals,
    resolve_intent,
)
from oce.domain.services.query_classifier import QueryIntent, extract_hard_signals


def test_hard_extractor_captures_symbol_and_path_facts_without_network():
    signals = extract_hard_signals("`SearchQueryHandler` 在 src/search.py 中定义")

    assert signals.has_concrete_symbol is True
    assert signals.symbol_tokens == ("SearchQueryHandler",)
    assert signals.has_filename is True
    assert signals.has_explicit_path is True
    assert signals.has_definition_marker is True


def test_hard_extractor_only_marks_real_compound_requests():
    assert extract_hard_signals("架构和事件处理机制").has_independent_clauses is False
    signals = extract_hard_signals("找出配置文件，并说明启动时如何加载它")
    assert signals.clause_count == 2
    assert signals.has_independent_clauses is True


def test_symbol_hard_rule_cannot_become_path_from_filename():
    decision = resolve_intent("`SearchQueryHandler` 定义在哪个 .py 文件？")
    assert decision.intent is QueryIntent.SYMBOL
    assert decision.reason == "concrete_symbol_default"


def test_flow_and_single_point_usage_precedence_for_concrete_symbol():
    assert resolve_intent("前端如何调用后端的 `handler`？").intent is QueryIntent.CALL_CHAIN
    assert resolve_intent("如何使用 `handler`？").intent is QueryIntent.REFERENCE


def test_hybrid_v2_maps_static_reference_scope_to_reference():
    decision = resolve_intent("In which files is `get_strategy` used?")
    assert decision.intent is QueryIntent.REFERENCE
    assert decision.reason == "static_reference_scope"
    assert decision.resolver_version == "hybrid-v2"


def test_frozen_benchmark_static_reference_convention_is_explicit_only():
    decision = resolve_intent(
        "In which files is `get_strategy` used?",
        legacy_static_reference_as_symbol=True,
    )
    assert decision.intent is QueryIntent.SYMBOL
    assert decision.reason == "concrete_symbol_default"


def test_soft_signal_confidence_is_gated_and_never_overrides_hard_constraints():
    path = extract_hard_signals("config.yaml 在哪里？")
    low_confidence = SoftSignals(
        asks_implementation=SoftSignal(True, confidence=0.2),
        asks_overview=SoftSignal(True, confidence=0.2),
    )
    assert (
        IntentResolver().resolve(
            "config.yaml 在哪里?", hard=path, soft=low_confidence
        ).intent
        is QueryIntent.PATH
    )

    symbol = extract_hard_signals("`handler`")
    usage = SoftSignals(asks_api_usage=True)
    assert (
        IntentResolver().resolve("`handler`", hard=symbol, soft=usage).intent
        is QueryIntent.REFERENCE
    )


def test_compound_requires_structural_evidence():
    decision = resolve_intent("找出登录实现，并说明密码重置逻辑")
    assert decision.intent is QueryIntent.COMPOUND
    assert resolve_intent("架构和事件处理机制").intent is QueryIntent.OVERVIEW


def test_flow_continuation_is_not_compound():
    decision = resolve_intent("How is `on_message` triggered and what does it call next?")
    assert decision.intent is QueryIntent.CALL_CHAIN


def test_independent_goals_outrank_incidental_path_or_symbol_cues():
    decision = resolve_intent(
        "Locate `AuditLogger` implementation and separately trace how it is reached."
    )
    assert decision.intent is QueryIntent.COMPOUND


def test_decision_is_auditable_and_provider_free_by_default():
    decision = resolve_intent("这是什么项目？")
    assert isinstance(decision, IntentDecision)
    assert decision.used_laya is False
    assert decision.hard_signals is not None
    assert decision.soft_signals is not None
    assert decision.resolver_version == "hybrid-v2"


def test_decision_snapshot_is_json_serializable():
    decision = resolve_intent(
        "How is `handler` used?",
        soft=SoftSignals(asks_api_usage=SoftSignal(True, confidence=0.91)),
    )
    encoded = json.dumps(decision.as_dict(), ensure_ascii=False)
    assert '"value": true' in encoded
