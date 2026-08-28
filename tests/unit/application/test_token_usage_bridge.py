"""容器 token 用量桥接：把 embedder/reranker/llm 的回调映射成 TokenUsageRecord。

只验证纯映射逻辑（credential_id=0 归一 None、total=prompt+completion），无需构造
完整 Container——直接以裸对象充当 self 调用未绑定方法即可。
"""
from __future__ import annotations

from types import SimpleNamespace

from oce.application.container import Container
from oce.shared.metrics import TokenUsageRecord


class _RecordingSink:
    def __init__(self) -> None:
        self.records: list[TokenUsageRecord] = []

    def record_token_usage(self, record: TokenUsageRecord) -> None:
        self.records.append(record)


async def test_bridge_maps_usage_and_normalizes_zero_credential():
    sink = _RecordingSink()
    holder = SimpleNamespace(metrics=sink)

    # LLM：credential_id=0 → None，total = 12 + 5
    await Container._record_token_usage(holder, 0, "llm", "m", 12, 5)
    rec = sink.records[0]
    assert rec.kind == "llm"
    assert rec.model == "m"
    assert rec.total_tokens == 17
    assert rec.credential_id is None

    # embed：真实凭证 id 透传，completion=0
    await Container._record_token_usage(holder, 7, "embed", "e", 10, 0)
    assert sink.records[1].credential_id == 7
    assert sink.records[1].total_tokens == 10


async def test_bridge_swallows_sink_errors():
    """旁路容错：sink 抛错也不冒泡回主链路。"""

    class _BoomSink:
        def record_token_usage(self, record: TokenUsageRecord) -> None:
            raise RuntimeError("boom")

    holder = SimpleNamespace(metrics=_BoomSink())
    # 不抛异常即通过
    await Container._record_token_usage(holder, 1, "rerank", "m", 3, 0)
