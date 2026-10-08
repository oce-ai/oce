"""判定表：优先级即数据。

重构前判定优先级藏在嵌套 if 的缩进里，两个 repo 的缩进顺序已经不一致。
这里把它写成一条有序规则列表：按序求值，返回首个命中。好处是优先级可以
被测试直接断言，也能在报告里按 reason 统计分布。

规则顺序的依据（从高到低）：

1. 空查询 —— 结构事实，无需语义。
2. 独立子句 —— 结构事实，压过任一子句内部的符号/路径线索。
3. 具体符号分支 —— 符号压过文件名措辞（「`parse_config` 在 server.py 里
   注册了哪些路由」问的是符号，不是文件）。分支内部：
   flow > usage(U) > reference(R) > definition(S) > 裸符号默认 S。
   usage 先于 reference 是因为「哪些文件用到它」比「怎么调用它」结构更明确；
   两者同现时（「它在哪些文件用到，参数是什么」）会先被独立子句规则捕获。
4. 显式路径 —— 仅当没有功能/架构语义在竞争时才成立。
5. 架构语义 -> O。
6. 功能语义 -> F。
7. 兜底 -> F。

规则只读 `Signals`，不含任何 benchmark ID 或具体查询文本。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from oce.domain.services.intent.signals import Signals
from oce.domain.services.intent.taxonomy import QueryIntent


@dataclass(frozen=True, slots=True)
class Rule:
    """一条判定规则：条件 + 结论 + 稳定的 reason 标识。"""

    reason: str
    intent: QueryIntent
    condition: Callable[[Signals], bool]
    #: 该规则是否为结构性硬事实。硬事实不接受概率源覆盖。
    hard: bool = True


#: 有序判定表。`resolve` 按序求值并返回首个命中的规则。
RULES: tuple[Rule, ...] = (
    Rule(
        reason="empty_query",
        intent=QueryIntent.FEATURE,
        condition=lambda s: s.is_empty,
    ),
    # 多个独立检索目标是结构事实，压过任一子句内的符号/路径线索。
    Rule(
        reason="independent_clauses",
        intent=QueryIntent.COMPOUND,
        condition=lambda s: s.has_independent_clauses,
    ),
    # ── 具体符号分支：符号压过文件名措辞 ──────────────────────────────
    Rule(
        reason="symbol_call_chain",
        intent=QueryIntent.CALL_CHAIN,
        condition=lambda s: s.has_symbol and s.has_flow,
    ),
    Rule(
        reason="symbol_usage_sites",
        intent=QueryIntent.USAGE,
        condition=lambda s: s.has_symbol and s.has_usage,
    ),
    Rule(
        reason="symbol_api_contract",
        intent=QueryIntent.REFERENCE,
        condition=lambda s: s.has_symbol and s.has_reference,
    ),
    Rule(
        reason="symbol_definition",
        intent=QueryIntent.SYMBOL,
        condition=lambda s: s.has_symbol and s.has_definition,
    ),
    # 裸符号无其他线索时按分类法取定义查找（S 是符号分支的默认语义）。
    Rule(
        reason="bare_symbol_default",
        intent=QueryIntent.SYMBOL,
        condition=lambda s: s.has_symbol,
        hard=False,
    ),
    # ── 无符号分支 ────────────────────────────────────────────────────
    # 无符号但问「哪里用到某功能」仍是引用点语义。
    Rule(
        reason="usage_sites_without_symbol",
        intent=QueryIntent.USAGE,
        condition=lambda s: s.has_usage and not s.has_overview and not s.has_feature,
        hard=False,
    ),
    Rule(
        reason="flow_without_symbol",
        intent=QueryIntent.CALL_CHAIN,
        condition=lambda s: s.has_flow and not s.has_overview,
        hard=False,
    ),
    # 路径只在没有功能/架构语义竞争时成立。「哪个文件实现了分块逻辑」问的是
    # 实现（F），不是文件本身（P）。
    Rule(
        reason="explicit_path_lookup",
        intent=QueryIntent.PATH,
        condition=lambda s: s.has_path and not s.has_feature and not s.has_overview,
    ),
    Rule(
        reason="system_overview",
        intent=QueryIntent.OVERVIEW,
        condition=lambda s: s.has_overview,
        hard=False,
    ),
    Rule(
        reason="feature_implementation",
        intent=QueryIntent.FEATURE,
        condition=lambda s: s.has_feature,
        hard=False,
    ),
    # 兜底：自然语言描述且无任何结构线索时按功能定位处理。
    Rule(
        reason="conservative_default",
        intent=QueryIntent.FEATURE,
        condition=lambda s: True,
        hard=False,
    ),
)


def match(signals: Signals) -> Rule:
    """返回首个命中的规则。判定表末条恒为真，故必有返回。"""
    for rule in RULES:
        if rule.condition(signals):
            return rule
    # 防御性：末条规则恒真，正常不会到这里。
    return RULES[-1]
