"""概率判定源的抽象接口（域层侧）。

域层只认这个 Protocol，不 import 任何 HTTP 客户端或 SDK。具体实现放在
`oce.infrastructure.intent`，由 composition root 注入。这样换供应商、换
传输方式都不会触碰判定逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from oce.domain.services.intent.taxonomy import QueryIntent


@dataclass(frozen=True, slots=True)
class IntentPrediction:
    """概率判定源返回的一次预测。

    `confidence` 直接来自供应商（TypeSafe Choice 答案自带 confidence），
    不由本地概率再推导，避免两处口径不一致。
    """

    intent: QueryIntent
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    model: str | None = None
    latency_ms: float | None = None


class IntentProviderError(RuntimeError):
    """判定源不可用。调用方必须据此降级，而不是把异常抛给用户。

    `reason` 是稳定的短标识（如 ``timeout``、``http_429``），进审计记录用。
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@runtime_checkable
class IntentProvider(Protocol):
    """把一条查询判定为意图的概率源。"""

    async def predict(self, query: str) -> IntentPrediction:
        """返回预测；任何失败都必须抛 `IntentProviderError`。"""
        ...
