"""判定表与信号提取的单元测试。

重点在两件事：
1. 优先级是可断言的数据（而非藏在缩进里）；
2. 8 个标签各自的边界，特别是新增的 U 与 S / R 的区别。
"""

from __future__ import annotations

import pytest

from oce.domain.services.intent.resolver import resolve_rules
from oce.domain.services.intent.rules import RULES, match
from oce.domain.services.intent.signals import (
    extract_signals,
    extract_symbols,
    split_independent_clauses,
)
from oce.domain.services.intent.taxonomy import LABELS, QueryIntent
from oce.domain.services.intent.signals import Signals


# ── 判定表结构 ──────────────────────────────────────────────────────────────


def test_rule_table_reasons_are_unique_and_stable() -> None:
    reasons = [rule.reason for rule in RULES]
    assert len(reasons) == len(set(reasons)), "reason 必须唯一，否则报告无法归因"


def test_rule_table_last_rule_is_total() -> None:
    """末条规则必须恒真，保证 match 总有返回。"""
    assert RULES[-1].condition(Signals()) is True
    assert RULES[-1].intent is QueryIntent.FEATURE


def test_priority_order_is_explicit_data() -> None:
    """直接断言优先级顺序，而不是间接靠查询文本推断。"""
    order = [rule.reason for rule in RULES]
    expected_prefix = [
        "empty_query",
        "independent_clauses",
        "symbol_call_chain",
        "symbol_usage_sites",
        "symbol_api_contract",
        "symbol_definition",
        "bare_symbol_default",
    ]
    assert order[: len(expected_prefix)] == expected_prefix
    # 兜底必须在最后
    assert order[-1] == "conservative_default"


def test_independent_clauses_outrank_symbol_branch() -> None:
    """复合结构压过符号分支：两者同时为真时必须判 M。"""
    signals = Signals(
        has_symbol=True,
        symbols=("foo_bar",),
        has_definition=True,
        has_independent_clauses=True,
        clause_count=2,
    )
    assert match(signals).intent is QueryIntent.COMPOUND


def test_symbol_branch_internal_precedence() -> None:
    """符号分支内部：flow > usage > reference > definition。"""
    base = dict(has_symbol=True, symbols=("foo_bar",))
    assert match(Signals(**base, has_flow=True, has_usage=True)).intent is QueryIntent.CALL_CHAIN
    assert match(Signals(**base, has_usage=True, has_reference=True)).intent is QueryIntent.USAGE
    assert (
        match(Signals(**base, has_reference=True, has_definition=True)).intent
        is QueryIntent.REFERENCE
    )
    assert match(Signals(**base, has_definition=True)).intent is QueryIntent.SYMBOL
    assert match(Signals(**base)).intent is QueryIntent.SYMBOL


def test_path_yields_to_feature_and_overview() -> None:
    """路径只在没有功能/架构语义竞争时成立。"""
    assert match(Signals(has_path=True)).intent is QueryIntent.PATH
    assert match(Signals(has_path=True, has_feature=True)).intent is QueryIntent.FEATURE
    assert match(Signals(has_path=True, has_overview=True)).intent is QueryIntent.OVERVIEW


# ── 标签边界（端到端查询） ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "query,expected",
    [
        # S：定义/声明/源码位置
        ("`parse_config` 在哪里定义？", QueryIntent.SYMBOL),
        ("Where is `foo_bar` defined?", QueryIntent.SYMBOL),
        ("Which file defines `build_chunker`?", QueryIntent.SYMBOL),
        # U：引用点（新标签），与 S 和 R 都不同
        ("哪些文件引用了 `parse_config`？", QueryIntent.USAGE),
        ("In which files is `get_strategy` used?", QueryIntent.USAGE),
        ("`SearchStore.search` 的调用点都有哪些？", QueryIntent.USAGE),
        ("Find all callers of `save_order`", QueryIntent.USAGE),
        # R：API 契约
        ("`build_index` 的参数和返回值是什么？", QueryIntent.REFERENCE),
        ("How should callers invoke `questions_from_json`?", QueryIntent.REFERENCE),
        ("How do I call `build_index` and what does it return?", QueryIntent.REFERENCE),
        # C：调用链
        ("trace the call chain from the CLI entry to the index writer", QueryIntent.CALL_CHAIN),
        ("从 `api_search` 到 rerank 的完整 flow", QueryIntent.CALL_CHAIN),
        ("Trace `save_order` through validation, repository, and database.", QueryIntent.CALL_CHAIN),
        # P：路径
        ("config.json 在哪里？", QueryIntent.PATH),
        ("日志配置 yaml 文件在哪里？", QueryIntent.PATH),
        # F：功能实现
        ("登录失败重试的业务规则实现在哪里？", QueryIntent.FEATURE),
        ("在 src/oce/domain 目录下哪个文件实现了分块逻辑？", QueryIntent.FEATURE),
        ("Explain the retry behavior", QueryIntent.FEATURE),
        # O：架构
        ("检索结果的缓存一致性机制是怎么设计的？", QueryIntent.OVERVIEW),
        ("请求从前端到后端整体怎么交互？", QueryIntent.OVERVIEW),
        # M：复合
        ("找出 `parse_config` 的定义，并说明它的调用链路", QueryIntent.COMPOUND),
    ],
)
def test_label_boundaries(query: str, expected: QueryIntent) -> None:
    assert resolve_rules(query).intent is expected


def test_usage_symbol_and_reference_are_three_distinct_answers() -> None:
    """同一符号的三类问法必须落到三个不同标签。"""
    symbol = resolve_rules("`parse_config` 在哪里定义？").intent
    usage = resolve_rules("哪些文件引用了 `parse_config`？").intent
    reference = resolve_rules("`parse_config` 怎么调用，参数是什么？").intent
    assert {symbol, usage, reference} == {
        QueryIntent.SYMBOL,
        QueryIntent.USAGE,
        QueryIntent.REFERENCE,
    }


# ── 信号提取 ────────────────────────────────────────────────────────────────


def test_paths_and_filenames_are_not_symbols() -> None:
    """路径与文件名必须先被掩掉，不能漏成假符号。"""
    assert extract_symbols("src/module_008.yaml 在哪里？") == ()
    assert extract_symbols("package.json 里有什么？") == ()
    assert "README" not in extract_symbols("README.zh-CN.md 的内容")


def test_backtick_path_and_hyphen_are_not_symbols() -> None:
    assert extract_symbols("`src/oce/main.py`") == ()
    assert extract_symbols("`feature-028` 的实现") == ()


def test_symbols_are_extracted_in_order() -> None:
    assert extract_symbols("先看 `parse_config` 再看 `build_index`") == (
        "parse_config",
        "build_index",
    )


@pytest.mark.parametrize(
    "query,symbol,expected",
    [
        ("parse_config的定义在哪里？", "parse_config", QueryIntent.SYMBOL),
        ("请查看parse_config的定义", "parse_config", QueryIntent.SYMBOL),
        ("哪些文件引用了SearchStore.search？", "SearchStore.search", QueryIntent.USAGE),
        ("哪些文件引用了tauri::command？", "tauri::command", QueryIntent.USAGE),
        ("MyClass的定义在哪里？", "MyClass", QueryIntent.SYMBOL),
        ("请定位Provider的类型定义", "Provider", QueryIntent.SYMBOL),
        ("parse()的参数是什么？", "parse", QueryIntent.REFERENCE),
    ],
)
def test_symbols_adjacent_to_chinese_keep_exact_recall_anchors(
    query: str, symbol: str, expected: QueryIntent
) -> None:
    assert symbol in extract_symbols(query)
    assert resolve_rules(query).intent is expected


@pytest.mark.parametrize("symbol", ["MyClass", "ABHandler", "IOError", "XHandler"])
def test_short_camel_prefixes_are_concrete_definition_symbols(symbol: str) -> None:
    query = f"Where is {symbol} defined?"
    assert extract_symbols(query) == (symbol,)
    assert resolve_rules(query).intent is QueryIntent.SYMBOL


def test_identifier_boundaries_do_not_extract_suffixes_of_ascii_tokens() -> None:
    assert extract_symbols("Where is 1MyClass defined?") == ()
    assert extract_symbols("Where is 1parse_config defined?") == ()


def test_semantic_words_inside_backticks_do_not_fire() -> None:
    """``invoke_handler`` 里的 invoke 不应触发 flow/reference 信号。"""
    signals = extract_signals("`invoke_handler`")
    assert signals.has_symbol is True
    assert signals.has_flow is False
    assert signals.has_reference is False


def test_empty_query_is_flagged() -> None:
    assert extract_signals("").is_empty is True
    assert extract_signals("   ").is_empty is True
    assert extract_signals("?!").is_empty is True
    assert resolve_rules("").reason == "empty_query"


def test_signals_snapshot_is_serializable() -> None:
    snapshot = extract_signals("`foo_bar` 在哪里定义？").as_dict()
    assert snapshot["has_symbol"] is True
    assert isinstance(snapshot["symbols"], list)
    import json

    json.dumps(snapshot)  # 不抛异常即可进审计记录


# ── 复合拆分 ────────────────────────────────────────────────────────────────


def test_conjunction_alone_is_not_compound() -> None:
    """「架构和事件处理」是一个目标，不是两个。"""
    assert resolve_rules("架构和事件处理是怎么设计的？").intent is not QueryIntent.COMPOUND
    assert split_independent_clauses("架构和事件处理是怎么设计的？") == [
        "架构和事件处理是怎么设计的？"
    ]


def test_call_chain_continuation_is_not_compound() -> None:
    query = "When `save_order` is triggered, what does it call next?"
    assert resolve_rules(query).intent is not QueryIntent.COMPOUND


def test_split_matches_detection() -> None:
    """检测判定与拆分行为必须一致。"""
    query = "找出 `parse_config` 的定义，并说明它的调用链路"
    assert resolve_rules(query).intent is QueryIntent.COMPOUND
    assert len(split_independent_clauses(query)) >= 2


@pytest.mark.parametrize(
    "query,expected_parts",
    [
        (
            "Where is `foo_bar` defined and what does it return?",
            ["Where is `foo_bar` defined", "what does it return"],
        ),
        (
            "Where is `foo_bar` defined and what does it call next?",
            ["Where is `foo_bar` defined", "what does it call next"],
        ),
        (
            "Where is `foo_bar` defined and how do I call it and what does it return?",
            ["Where is `foo_bar` defined", "how do I call it and what does it return"],
        ),
        (
            "How do I call `foo_bar` and what does it return, and locate config.json?",
            ["How do I call `foo_bar` and what does it return", "locate config.json"],
        ),
        (
            "Find config.json and trace `foo_bar` and what does it call next?",
            ["Find config.json", "trace `foo_bar` and what does it call next"],
        ),
        (
            "Trace `foo_bar` and what does it call next and locate config.json?",
            ["Trace `foo_bar` and what does it call next", "locate config.json"],
        ),
        (
            "说明架构和事件处理，并且定位config.json",
            ["说明架构和事件处理", "定位config.json"],
        ),
        (
            "Where is `foo_bar` defined? Where is config.json?",
            ["Where is `foo_bar` defined", "Where is config.json"],
        ),
        (
            "找出`parse_config`的定义；说明重试逻辑",
            ["找出`parse_config`的定义", "说明重试逻辑"],
        ),
        (
            "Find config.json\nExplain retry behavior",
            ["Find config.json", "Explain retry behavior"],
        ),
        (
            "Where is `foo_bar` defined? How do I call it and what does it return?",
            ["Where is `foo_bar` defined", "How do I call it and what does it return"],
        ),
    ],
)
def test_compound_detection_and_recall_use_the_same_scoped_boundaries(
    query: str, expected_parts: list[str]
) -> None:
    decision = resolve_rules(query)
    assert decision.intent is QueryIntent.COMPOUND
    assert decision.signals.clause_count == len(expected_parts)
    assert split_independent_clauses(query) == expected_parts


@pytest.mark.parametrize(
    "query,expected",
    [
        ("How do I call `foo_bar` and what does it return?", QueryIntent.REFERENCE),
        ("`foo_bar`怎么调用和它的返回值是什么？", QueryIntent.REFERENCE),
        ("Trace `foo_bar` and what does it call next?", QueryIntent.CALL_CHAIN),
        ("架构和事件处理是怎么设计的？", QueryIntent.OVERVIEW),
    ],
)
def test_single_goal_continuations_stay_in_one_recall_branch(
    query: str, expected: QueryIntent
) -> None:
    decision = resolve_rules(query)
    assert decision.intent is expected
    assert decision.signals.clause_count == 1
    assert split_independent_clauses(query) == [query]


# ── 决策记录 ────────────────────────────────────────────────────────────────


def test_decision_exposes_reason_and_source() -> None:
    decision = resolve_rules("`foo_bar` 在哪里定义？")
    assert decision.reason == "symbol_definition"
    assert decision.source == "rule_hard"
    assert decision.rule_intent is QueryIntent.SYMBOL
    assert decision.provider_intent is None


def test_all_labels_are_reachable_by_the_rule_table() -> None:
    """每个标签都必须至少有一条规则能产出它，否则它是死标签。"""
    produced = {rule.intent for rule in RULES}
    assert produced == set(LABELS), f"未被任何规则产出的标签: {set(LABELS) - produced}"
