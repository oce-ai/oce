"""查询分类的兼容入口。

判定逻辑已经全部迁到 `oce.domain.services.intent`（分类法 / 模式 / 信号 /
判定表 / 仲裁器）。本模块只保留少量对外仍在调用的同步入口，全部委托给那
份唯一实现，自己不再持有任何正则或优先级。

保留这些函数是为了不破坏既有调用方（检索层、路径索引判定）。
"""

from __future__ import annotations

from oce.domain.services.intent.resolver import resolve_rules
from oce.domain.services.intent.signals import (
    Signals,
    extract_signals,
    extract_symbols,
    split_independent_clauses,
)
from oce.domain.services.intent.taxonomy import QueryIntent


def classify_query_intent(query: str) -> QueryIntent:
    """纯规则意图判定（同步）。

    与 `IntentResolver` 的降级路径共用同一张判定表，因此两者对同一查询
    永远给出相同标签；`tests/unit/domain/test_intent_consistency.py` 对此
    有断言。

    Examples:
        >>> classify_query_intent("`parse_config` 函数在哪里定义？")
        <QueryIntent.SYMBOL: 'S'>

        >>> classify_query_intent("哪些文件引用了 `parse_config`？")
        <QueryIntent.USAGE: 'U'>

        >>> classify_query_intent("config.json 在哪里？")
        <QueryIntent.PATH: 'P'>
    """
    return resolve_rules(query).intent


def extract_code_identifiers(query: str) -> tuple[str, ...]:
    """提取代码标识符，保持出现顺序（委托给唯一实现）。"""
    return extract_symbols(query)


def has_code_identifier(query: str) -> bool:
    """查询是否包含代码标识符而非纯自然语言描述。"""
    return bool(extract_symbols(query))


def is_filename_query(query: str) -> tuple[bool, float]:
    """是否为文件名查询，返回 ``(判定, 置信度)``。

    Examples:
        >>> is_filename_query("主配置文件在哪里？")
        (True, 0.9)

        >>> is_filename_query("`parse_config` 函数在哪里？")
        (False, 0.0)
    """
    if classify_query_intent(query) is QueryIntent.PATH:
        return True, 0.9
    return False, 0.0


def should_use_path_index(query: str, threshold: float = 0.5) -> bool:
    """是否走路径索引。

    带符号锚点的查询即便含扩展名也判为符号意图，不路由到 path index，
    因为它要找的是符号而非文件本身。``threshold`` 仅为向后兼容保留。

    Examples:
        >>> should_use_path_index("config.json 在哪里？")
        True

        >>> should_use_path_index("`parse_config` 在 server.py 中注册了哪些路由？")
        False
    """
    return classify_query_intent(query) is QueryIntent.PATH
