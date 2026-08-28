"""监控指标端口。

采集侧（HTTP 中间件、embedder/reranker/llm client、资源采样器）只依赖本模块的
Protocol 与 record 数据结构，不关心落库细节；具体 sink 由 composition root 注入。

约束：``record_*`` 必须同步、非阻塞、绝不把异常抛回调用方——监控是旁路，任何情况下
都不能拖慢或中断主链路。
"""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import perf_counter
from typing import Protocol


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ApiCallRecord:
    endpoint: str
    method: str
    status_code: int
    latency_ms: int
    error_type: str | None = None
    ts: datetime = field(default_factory=_now)


@dataclass(frozen=True)
class TokenUsageRecord:
    kind: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    credential_id: int | None = None
    ts: datetime = field(default_factory=_now)


@dataclass(frozen=True)
class ResourceSampleRecord:
    disk_data_bytes: int
    disk_free_bytes: int
    disk_total_bytes: int
    mem_rss_bytes: int
    mem_percent: float
    cpu_percent: float
    ts: datetime = field(default_factory=_now)


@dataclass(frozen=True)
class RetrievalMetricRecord:
    """一次检索的审计上报：hit_count=0 即空回；stages 存各阶段耗时（毫秒）。"""

    source: str
    hit_count: int
    total_ms: int
    scope_size: int | None = None
    intent: str | None = None
    path_boosted: bool = False
    query_text: str | None = None
    stages: dict[str, int] = field(default_factory=dict)
    ts: datetime = field(default_factory=_now)


@dataclass
class RetrievalAudit:
    """检索管线阶段耗时的可变收集容器。

    domain 只负责填数据、不依赖 sink：各阶段用 ``with audit.stage("dense"):`` 计时，
    括住的代码块含 await 也能测出墙钟耗时；同名阶段累加（多子查询召回不会互相覆盖）。
    """

    intent: str | None = None
    path_boosted: bool = False
    scope_size: int | None = None
    stages: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = perf_counter()
        try:
            yield
        finally:
            elapsed = int((perf_counter() - start) * 1000)
            self.stages[name] = self.stages.get(name, 0) + elapsed


class MetricsSink(Protocol):
    """监控采集端口。实现方的 record_* 必须同步、非阻塞、不抛出。"""

    def record_api_call(self, record: ApiCallRecord) -> None: ...

    def record_token_usage(self, record: TokenUsageRecord) -> None: ...

    def record_resource_sample(self, record: ResourceSampleRecord) -> None: ...

    def record_retrieval(self, record: RetrievalMetricRecord) -> None: ...


class NoopMetricsSink:
    """监控关闭时的空实现（也满足生命周期接口，便于统一装配）。"""

    def record_api_call(self, record: ApiCallRecord) -> None:
        return None

    def record_token_usage(self, record: TokenUsageRecord) -> None:
        return None

    def record_resource_sample(self, record: ResourceSampleRecord) -> None:
        return None

    def record_retrieval(self, record: RetrievalMetricRecord) -> None:
        return None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None
