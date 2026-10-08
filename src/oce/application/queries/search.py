"""查询对象与处理器 - 检索读路径

SearchQuery: 一次代码检索（向量召回 + 精确标识符召回 + 重排 + 覆盖度选择）。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from time import perf_counter

from loguru import logger

from oce.application.messages import Query
from oce.domain.services.retrieval import RetrievalPipeline
from oce.domain.services.search import SearchHit
from oce.shared.metrics import (
    MetricsSink,
    NoopMetricsSink,
    RetrievalAudit,
    RetrievalMetricRecord,
)


@dataclass(frozen=True)
class SearchQuery(Query):
    """检索查询"""

    query: str
    allowed_blob_names: frozenset[str] | None = None
    source: str = "retrieval"


@dataclass(frozen=True)
class SearchResult:
    """检索结果"""

    hits: list[SearchHit] = field(default_factory=list)


class SearchQueryHandler:
    """处理 SearchQuery。

    检索审计开启时，为本次检索创建 RetrievalAudit 传入 pipeline 收集各阶段耗时，
    检索完成后按 source 上报（hit_count=0 即空回）。审计上报走旁路 sink，不影响主链路。
    """

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        *,
        metrics: MetricsSink | None = None,
        retrieval_audit_enabled: bool = False,
        store_query_text: bool = False,
        on_close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.metrics = metrics or NoopMetricsSink()
        self.retrieval_audit_enabled = retrieval_audit_enabled
        self.store_query_text = store_query_text
        self._on_close = on_close
        self._active = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._retired = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def closed(self) -> bool:
        return self._close_task is not None and self._close_task.done()

    async def retire(self) -> None:
        """换栈后停止持有连接；已经进入旧 handler 的请求仍用旧配置跑完。"""
        self._retired = True
        if self._active == 0:
            await self._close_once()

    async def aclose(self) -> None:
        """关闭容器时等待所有旧请求及其连接回收完成。"""
        self._retired = True
        await self._idle.wait()
        await self._close_once()

    async def _close_once(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_resources())
        await asyncio.shield(self._close_task)

    async def _close_resources(self) -> None:
        if self._on_close is not None:
            try:
                await self._on_close()
            except Exception as exc:
                # 清理失败不能把已完成的检索变成错误，也不能跳过其它依赖的关闭。
                logger.warning("retrieval resource close failed: {}", type(exc).__name__)

    async def handle(self, query: SearchQuery) -> SearchResult:
        # QueryBus 取 handler 与进入这里之间没有 await，计数先于任何检索 I/O，
        # 因此热替换可以可靠地区分已进入的旧请求与新请求。
        self._active += 1
        self._idle.clear()
        try:
            return await self._handle(query)
        finally:
            self._active -= 1
            if self._active == 0:
                self._idle.set()
                if self._retired:
                    await self._close_once()

    async def _handle(self, query: SearchQuery) -> SearchResult:
        if not self.retrieval_audit_enabled:
            hits = await self.pipeline.search(query.query, query.allowed_blob_names)
            return SearchResult(hits=hits)

        audit = RetrievalAudit()
        started = perf_counter()
        hits = await self.pipeline.search(
            query.query, query.allowed_blob_names, audit=audit
        )
        total_ms = int((perf_counter() - started) * 1000)
        self.metrics.record_retrieval(
            RetrievalMetricRecord(
                source=query.source,
                hit_count=len(hits),
                total_ms=total_ms,
                scope_size=audit.scope_size,
                intent=audit.intent,
                intent_source=audit.intent_source,
                intent_decision_reason=audit.intent_decision_reason,
                path_boosted=audit.path_boosted,
                query_text=query.query if self.store_query_text else None,
                stages=dict(audit.stages),
            )
        )
        return SearchResult(hits=hits)
