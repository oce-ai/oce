"""意图分类法：8 个标签的唯一真源。

标签存在的唯一标准：**它必须能导出一条与其他标签不同的检索策略**。
若两个标签的检索行为完全相同，它们就应该合并；`retrieval_strategy`
中有一个测试对此做断言。

与旧 7 类体系的差异：
- `S` 收窄为定义/声明/源码位置，不再吞掉静态引用查询；
- `R` 收窄为 API 契约语义（签名、参数、调用示例）；
- 新增 `U`（引用点）承接「哪些文件/位置用到了某符号」。这类查询要的是
  跨文件广度召回且不要定义加权，与 `S`（定义加权、少量块）和 `R`
  （签名与示例、权威少量位置）都不同，因此是独立标签而非 S/R 的别名。

`criteria` 文本同时用于两处，必须保持一致：
1. 作为 TypeSafe System One Choice 问题的 criteria 下发给模型；
2. 作为本文件的人类可读规格。
"""

from __future__ import annotations

from enum import StrEnum


class QueryIntent(StrEnum):
    """代码检索查询意图。

    值使用单字母代码，与回归数据集的 gold 标签直接对齐，省掉一层映射表。
    仓库中只允许存在这一个意图枚举定义。
    """

    SYMBOL = "S"
    CALL_CHAIN = "C"
    REFERENCE = "R"
    USAGE = "U"
    PATH = "P"
    FEATURE = "F"
    OVERVIEW = "O"
    COMPOUND = "M"


#: 标签的规范顺序，用于报告列顺序与 Choice criteria 的下发顺序。
LABELS: tuple[QueryIntent, ...] = (
    QueryIntent.SYMBOL,
    QueryIntent.CALL_CHAIN,
    QueryIntent.REFERENCE,
    QueryIntent.USAGE,
    QueryIntent.PATH,
    QueryIntent.FEATURE,
    QueryIntent.OVERVIEW,
    QueryIntent.COMPOUND,
)

#: 每个标签的判定说明。作为 Choice 的 criteria 下发，也是人类可读规格。
CRITERIA: dict[QueryIntent, str] = {
    QueryIntent.SYMBOL: (
        "Where a concrete code symbol is defined or declared: its definition "
        "site, declaration, source text, or registration point."
    ),
    QueryIntent.CALL_CHAIN: (
        "A multi-step execution or call path across boundaries: what calls "
        "what, in what order, end to end."
    ),
    QueryIntent.REFERENCE: (
        "How to call or use a concrete symbol: its signature, parameters, "
        "return value, or a usage example. The API contract itself."
    ),
    QueryIntent.USAGE: (
        "Where a concrete symbol is referenced or called from: which files, "
        "call sites, or places in the codebase use it. Not its definition and "
        "not its signature."
    ),
    QueryIntent.PATH: (
        "Locating a file, directory, configuration file, or filesystem path "
        "by name or extension."
    ),
    QueryIntent.FEATURE: (
        "Where or how a feature, behavior, or business rule is implemented, "
        "described without naming a concrete code symbol."
    ),
    QueryIntent.OVERVIEW: (
        "System-level design: architecture, mechanism, scheduling, state "
        "management, cross-layer interaction, or overall data flow."
    ),
    QueryIntent.COMPOUND: (
        "One query carrying several independent retrieval goals that each "
        "need their own lookup."
    ),
}

#: Choice 问题的 instructions。
INSTRUCTIONS = "Classify the retrieval intent of this code-search query."

#: 相邻标签对，评测报告用它高亮最容易混淆的边界。
BOUNDARY_PAIRS: tuple[tuple[QueryIntent, QueryIntent], ...] = (
    (QueryIntent.SYMBOL, QueryIntent.REFERENCE),
    (QueryIntent.SYMBOL, QueryIntent.USAGE),
    (QueryIntent.REFERENCE, QueryIntent.USAGE),
    (QueryIntent.REFERENCE, QueryIntent.CALL_CHAIN),
    (QueryIntent.USAGE, QueryIntent.CALL_CHAIN),
    (QueryIntent.SYMBOL, QueryIntent.PATH),
    (QueryIntent.FEATURE, QueryIntent.OVERVIEW),
    (QueryIntent.FEATURE, QueryIntent.COMPOUND),
    (QueryIntent.OVERVIEW, QueryIntent.COMPOUND),
)


def intent_from_label(label: str) -> QueryIntent | None:
    """把单字母 gold 标签解析为意图；无法识别时返回 ``None``。"""
    try:
        return QueryIntent((label or "").strip().upper())
    except ValueError:
        return None


def choice_criteria() -> dict[str, str]:
    """按规范顺序构造 Choice 的 criteria 映射（标签代码 -> 说明）。"""
    return {label.value: CRITERIA[label] for label in LABELS}
