"""监控指标的异步落库 sink：内存缓冲 + 后台批量 flush。

写库失败只记日志、丢弃该批，不重试也不上抛——监控是旁路，绝不影响主链路。
"""
from __future__ import annotations

import asyncio
from collections import deque
from typing import Callable

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from oce.infrastructure.persistence.models import (
    ApiCallMetricModel,
    ResourceSampleModel,
    RetrievalMetricModel,
    TokenUsageMetricModel,
)
from oce.shared.metrics import (
    ApiCallRecord,
    ResourceSampleRecord,
    RetrievalMetricRecord,
    TokenUsageRecord,
)


class SqlMetricsSink:
    """采集到的指标先入内存 deque，后台协程按间隔批量写库。

    ``deque(maxlen)`` 满时自动丢弃最旧样本，防止事件循环阻塞时缓冲无界增长。
    """

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        *,
        flush_interval_seconds: float = 5.0,
        max_buffer: int = 500,
    ) -> None:
        self._session_factory = session_factory
        self._flush_interval = flush_interval_seconds
        self._api: deque[ApiCallRecord] = deque(maxlen=max_buffer)
        self._token: deque[TokenUsageRecord] = deque(maxlen=max_buffer)
        self._resource: deque[ResourceSampleRecord] = deque(maxlen=max_buffer)
        self._retrieval: deque[RetrievalMetricRecord] = deque(maxlen=max_buffer)
        self._task: asyncio.Task | None = None
        self._running = False

    def record_api_call(self, record: ApiCallRecord) -> None:
        self._api.append(record)

    def record_token_usage(self, record: TokenUsageRecord) -> None:
        self._token.append(record)

    def record_resource_sample(self, record: ResourceSampleRecord) -> None:
        self._resource.append(record)

    def record_retrieval(self, record: RetrievalMetricRecord) -> None:
        self._retrieval.append(record)

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
            self._task = None
        await self._flush_once()

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_once()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("metrics flush loop error: {}", exc)

    def _drain(self) -> list:
        """原子取出三类缓冲并清空，构造成待插入的 ORM 行。"""
        rows: list = []
        while self._api:
            r = self._api.popleft()
            rows.append(
                ApiCallMetricModel(
                    ts=r.ts,
                    endpoint=r.endpoint,
                    method=r.method,
                    status_code=r.status_code,
                    latency_ms=r.latency_ms,
                    error_type=r.error_type,
                )
            )
        while self._token:
            r = self._token.popleft()
            rows.append(
                TokenUsageMetricModel(
                    ts=r.ts,
                    kind=r.kind,
                    model=r.model,
                    credential_id=r.credential_id,
                    prompt_tokens=r.prompt_tokens,
                    completion_tokens=r.completion_tokens,
                    total_tokens=r.total_tokens,
                )
            )
        while self._resource:
            r = self._resource.popleft()
            rows.append(
                ResourceSampleModel(
                    ts=r.ts,
                    disk_data_bytes=r.disk_data_bytes,
                    disk_free_bytes=r.disk_free_bytes,
                    disk_total_bytes=r.disk_total_bytes,
                    mem_rss_bytes=r.mem_rss_bytes,
                    mem_percent=r.mem_percent,
                    cpu_percent=r.cpu_percent,
                )
            )
        while self._retrieval:
            r = self._retrieval.popleft()
            s = r.stages
            rows.append(
                RetrievalMetricModel(
                    ts=r.ts,
                    source=r.source,
                    scope_size=r.scope_size,
                    hit_count=r.hit_count,
                    total_ms=r.total_ms,
                    intent=r.intent,
                    path_boosted=r.path_boosted,
                    query_text=r.query_text,
                    intent_ms=s.get("intent"),
                    rewrite_ms=s.get("rewrite"),
                    dense_ms=s.get("dense"),
                    exact_ms=s.get("exact"),
                    fuse_ms=s.get("fuse"),
                    rerank_ms=s.get("rerank"),
                    llm_rerank_ms=s.get("llm_rerank"),
                    select_ms=s.get("select"),
                )
            )
        return rows

    async def _flush_once(self) -> None:
        rows = self._drain()
        if not rows:
            return
        try:
            async with self._session_factory() as session:
                session.add_all(rows)
                await session.commit()
        except Exception as exc:
            logger.warning("metrics flush failed, dropped {} rows: {}", len(rows), exc)
