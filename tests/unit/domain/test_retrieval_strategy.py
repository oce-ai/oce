"""策略表测试：标签体系的自证。

核心断言：任意两个意图的策略配置不完全相同。若两行相同，说明这两个标签在
检索层没有区别，应该合并而不是并存。这条测试是 8 标签体系的存在性证明。
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import combinations

from oce.domain.services.intent.taxonomy import LABELS, QueryIntent
from oce.domain.services.retrieval_strategy import (
    STRATEGY_TABLE,
    RetrievalStrategy,
    get_strategy,
)


def test_every_label_has_a_strategy() -> None:
    assert set(STRATEGY_TABLE) == set(LABELS)
    assert len(STRATEGY_TABLE) == 8


def test_no_two_intents_share_the_same_strategy() -> None:
    """标签存在的唯一理由是它能导出不同的检索行为。

    重构前 FEATURE 与 COMPOUND 的配置一字不差，这条测试会直接抓到那种情况。
    """
    duplicates = []
    for left, right in combinations(sorted(STRATEGY_TABLE, key=lambda i: i.value), 2):
        if asdict(STRATEGY_TABLE[left]) == asdict(STRATEGY_TABLE[right]):
            duplicates.append(f"{left.value}=={right.value}")
    assert not duplicates, (
        f"以下意图的策略完全相同，它们在检索层无区别，应合并或差异化: {duplicates}"
    )


def test_compound_splits_clauses_and_feature_does_not() -> None:
    """M 区别于 F 的实质：按子句多路召回。"""
    assert STRATEGY_TABLE[QueryIntent.COMPOUND].split_clauses is True
    assert STRATEGY_TABLE[QueryIntent.FEATURE].split_clauses is False


def test_only_compound_splits_clauses() -> None:
    splitters = [i.value for i, s in STRATEGY_TABLE.items() if s.split_clauses]
    assert splitters == ["M"]


def test_usage_prefers_breadth_without_definition_boost() -> None:
    """U 区别于 S / R 的实质：跨文件铺开、不加权定义、块数放宽。"""
    usage = STRATEGY_TABLE[QueryIntent.USAGE]
    symbol = STRATEGY_TABLE[QueryIntent.SYMBOL]
    reference = STRATEGY_TABLE[QueryIntent.REFERENCE]

    assert usage.prefer_breadth is True
    assert usage.boost_definitions is False
    assert symbol.boost_definitions is True
    assert usage.max_chunks_per_path > symbol.max_chunks_per_path
    assert usage.max_chunks_per_path > reference.max_chunks_per_path


def test_only_usage_prefers_breadth() -> None:
    breadth = [i.value for i, s in STRATEGY_TABLE.items() if s.prefer_breadth]
    assert breadth == ["U"]


def test_only_path_enables_path_index() -> None:
    path_index = [i.value for i, s in STRATEGY_TABLE.items() if s.enable_path_index]
    assert path_index == ["P"]


def test_get_strategy_returns_table_entries() -> None:
    for intent in LABELS:
        assert get_strategy(intent) is STRATEGY_TABLE[intent]


def test_strategy_is_immutable() -> None:
    """策略是冻结的：调用方不能就地改掉共享的表项。"""
    import pytest

    strategy = get_strategy(QueryIntent.SYMBOL)
    with pytest.raises((AttributeError, TypeError)):
        strategy.max_chunks_per_path = 99  # type: ignore[misc]


def test_defaults_are_conservative() -> None:
    blank = RetrievalStrategy()
    assert blank.enable_path_index is False
    assert blank.split_clauses is False
    assert blank.prefer_breadth is False
