"""LLM 调用的 TPM（tokens per minute）限流。

这类 OpenAI 兼容网关按 60 秒滚动窗口统计 token，超限返回 429（如 SiliconFlow code=50602）。rerank 单次
请求可达 16k token，60k TPM 只够 3~4 次调用，必须在客户端排队而非事后重试：
重试失败会让 reranker 回退到原始顺序，评测结果混入未重排的查询而不易察觉。
"""

from __future__ import annotations

import asyncio
import time
from collections import deque


def estimate_tokens(text: str) -> int:
    """粗估 token 数，宁可高估以免触发 429。

    中日韩字符约 1 token/字，其余（代码、英文、符号）约 3 字符/token。
    """
    cjk = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff":
            cjk += 1
    other = len(text) - cjk
    return cjk + other // 3 + 1


class TokenRateLimiter:
    """滑动窗口 TPM 限流器，跨协程共享。"""

    def __init__(
        self,
        tokens_per_minute: int,
        window_seconds: float = 60.0,
        safety_ratio: float = 0.9,
    ) -> None:
        """
        Args:
            tokens_per_minute: 接口 TPM 上限
            window_seconds: 统计窗口长度
            safety_ratio: 预算折扣，留出估算误差余量
        """
        self.budget = max(1, int(tokens_per_minute * safety_ratio))
        self.window = window_seconds
        self._events: deque[tuple[float, int]] = deque()
        self._used = 0
        self._lock = asyncio.Lock()

    def _evict(self, now: float) -> None:
        """移出已滑出窗口的记账。"""
        while self._events and now - self._events[0][0] >= self.window:
            _, tokens = self._events.popleft()
            self._used -= tokens

    async def acquire(self, tokens: int) -> float:
        """申请额度，不足则等待窗口滑动。返回累计等待秒数。"""
        # 单请求超过整窗预算时按预算记账，否则永远等不到额度
        need = min(max(tokens, 1), self.budget)
        waited = 0.0

        while True:
            async with self._lock:
                now = time.monotonic()
                self._evict(now)
                if self._used + need <= self.budget:
                    self._events.append((now, need))
                    self._used += need
                    return waited
                # 最早一笔记账滑出窗口后才可能腾出额度
                sleep_for = self.window - (now - self._events[0][0])

            sleep_for = max(sleep_for, 0.05)
            waited += sleep_for
            await asyncio.sleep(sleep_for)
