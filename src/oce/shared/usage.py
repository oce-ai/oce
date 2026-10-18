"""请求级 token 计费账本。

客户端在构造期就把 usage 回调绑在单例上了（见 Container），没法按请求传收集器，
所以用 contextvar 把「当前请求的账本」透传下去：回调除了写监控 sink，再记一笔账，
请求返回时取总数即可。并发请求各自在自己的 task context 里，互不影响。

计价规则（面向下游计费）：
- embed / rerank 是纯输入型用量，直接相加；
- 其余（llm / llm_rerank / query_rewrite / intent）按 input + 4 * output 折算。
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

# 纯输入型 kind：completion 恒为 0，直接相加。其余一律按 LLM 折算。
_FLAT_KINDS = frozenset({"embed", "rerank"})
_OUTPUT_WEIGHT = 4


@dataclass
class UsageLedger:
    total_tokens: int = 0

    def add(self, kind: str, prompt_tokens: int, completion_tokens: int) -> None:
        weight = 1 if kind in _FLAT_KINDS else _OUTPUT_WEIGHT
        self.total_tokens += prompt_tokens + weight * completion_tokens


_current: ContextVar[UsageLedger | None] = ContextVar("oce_usage_ledger", default=None)


@contextmanager
def collect_usage() -> Iterator[UsageLedger]:
    """在 with 块内为当前请求开一本账；退出后账本仍可读取。"""
    ledger = UsageLedger()
    token = _current.set(ledger)
    try:
        yield ledger
    finally:
        _current.reset(token)


def record_usage(kind: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    """记一笔用量；不在 collect_usage() 作用域内时是空操作。"""
    ledger = _current.get()
    if ledger is not None:
        ledger.add(kind, prompt_tokens, completion_tokens)
