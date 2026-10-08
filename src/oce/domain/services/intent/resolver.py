"""仲裁器：硬事实 -> 判定表 -> 概率源。

四种判定来源，可在审计记录里区分：

- ``rule_hard``   结构性硬事实命中，不询问概率源；
- ``rule_only``   概率源缺席或未配置，用判定表的软结论；
- ``provider``    概率源给出高置信判定，被采纳；
- ``fallback``    概率源不可用或低置信，退回判定表结论。

概率源只在判定表给出**软**结论（`Rule.hard=False`）时才被询问。硬事实
（独立子句、符号+明确语义、显式路径）不交给模型，这既省调用也避免模型
推翻可验证的结构事实。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from oce.domain.services.intent.rules import match
from oce.domain.services.intent.port import (
    IntentProvider,
    IntentProviderError,
)
from oce.domain.services.intent.signals import Signals, extract_signals
from oce.domain.services.intent.taxonomy import QueryIntent

#: 判定来源标识。
SOURCE_RULE_HARD = "rule_hard"
SOURCE_RULE_ONLY = "rule_only"
SOURCE_PROVIDER = "provider"
SOURCE_FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class IntentDecision:
    """一次完整的、可审计的意图判定。"""

    intent: QueryIntent
    reason: str
    source: str
    signals: Signals
    #: 判定表给出的结论，始终存在（即便最终采纳了概率源）。
    rule_intent: QueryIntent | None = None
    #: 概率源的结论，仅在真正调用且成功时存在。
    provider_intent: QueryIntent | None = None
    provider_confidence: float | None = None
    provider_model: str | None = None
    provider_latency_ms: float | None = None
    #: 降级原因短标识；未降级时为 None。
    fallback_reason: str | None = None
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def used_provider(self) -> bool:
        return self.provider_intent is not None

    def as_dict(self) -> dict[str, Any]:
        """可序列化审计快照。"""
        return {
            "intent": self.intent.value,
            "reason": self.reason,
            "source": self.source,
            "rule_intent": self.rule_intent.value if self.rule_intent else None,
            "provider_intent": (
                self.provider_intent.value if self.provider_intent else None
            ),
            "provider_confidence": self.provider_confidence,
            "provider_model": self.provider_model,
            "provider_latency_ms": self.provider_latency_ms,
            "fallback_reason": self.fallback_reason,
            "probabilities": dict(self.probabilities),
            "signals": self.signals.as_dict(),
        }


def resolve_rules(query: str, signals: Signals | None = None) -> IntentDecision:
    """纯规则判定，绝不发起任何外部调用。

    这是离线评测与降级路径共用的入口，也是跨实现一致性测试的基准。
    """
    facts = signals if signals is not None else extract_signals(query)
    rule = match(facts)
    return IntentDecision(
        intent=rule.intent,
        reason=rule.reason,
        source=SOURCE_RULE_HARD if rule.hard else SOURCE_RULE_ONLY,
        signals=facts,
        rule_intent=rule.intent,
    )


class IntentResolver:
    """意图判定入口。注入 `provider` 才会咨询概率源。"""

    def __init__(
        self,
        *,
        provider: IntentProvider | None = None,
        min_confidence: float = 0.60,
    ) -> None:
        self.provider = provider
        self.min_confidence = min_confidence

    async def resolve(self, query: str) -> IntentDecision:
        facts = extract_signals(query)
        rule = match(facts)
        base = IntentDecision(
            intent=rule.intent,
            reason=rule.reason,
            source=SOURCE_RULE_HARD if rule.hard else SOURCE_RULE_ONLY,
            signals=facts,
            rule_intent=rule.intent,
        )

        # 硬事实不询问概率源：结构可验证，模型无权推翻。
        if rule.hard or self.provider is None:
            return base

        try:
            prediction = await self.provider.predict(query)
        except IntentProviderError as exc:
            return IntentDecision(
                intent=rule.intent,
                reason=rule.reason,
                source=SOURCE_FALLBACK,
                signals=facts,
                rule_intent=rule.intent,
                fallback_reason=exc.reason,
            )
        except Exception as exc:  # 适配器未包装的意外异常也必须降级
            return IntentDecision(
                intent=rule.intent,
                reason=rule.reason,
                source=SOURCE_FALLBACK,
                signals=facts,
                rule_intent=rule.intent,
                fallback_reason=f"unexpected_{type(exc).__name__}",
            )

        # 低置信不采纳：判定表的结论更保守，但可解释。
        if prediction.confidence < self.min_confidence:
            return IntentDecision(
                intent=rule.intent,
                reason=rule.reason,
                source=SOURCE_FALLBACK,
                signals=facts,
                rule_intent=rule.intent,
                provider_intent=prediction.intent,
                provider_confidence=prediction.confidence,
                provider_model=prediction.model,
                provider_latency_ms=prediction.latency_ms,
                fallback_reason="low_confidence",
                probabilities=dict(prediction.probabilities),
            )

        agrees = prediction.intent is rule.intent
        return IntentDecision(
            intent=prediction.intent,
            reason=f"{rule.reason};provider_{'agree' if agrees else 'override'}",
            source=SOURCE_PROVIDER,
            signals=facts,
            rule_intent=rule.intent,
            provider_intent=prediction.intent,
            provider_confidence=prediction.confidence,
            provider_model=prediction.model,
            provider_latency_ms=prediction.latency_ms,
            probabilities=dict(prediction.probabilities),
        )

    async def classify(self, query: str) -> QueryIntent:
        return (await self.resolve(query)).intent
