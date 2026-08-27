"""EmbedWorker — 消费嵌入队列，对 blob 补算向量

流程
----
    dequeue(blob_name) → IndexingPipeline.embed_pending([blob_name])
    成功 → ack；异常 → fail + DB retry_count++，超限置 error

并发
----
启动 N 个 worker 协程并行消费（concurrency 可配）。
每个协程一个消费循环，stop() 置标志后协程在下次 dequeue 超时自然退出。
每条消息在自己的 UoW 内构造独立 IndexingPipeline，协程间不共享可变状态。
"""
from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from loguru import logger

from oce.domain.services.indexing import IndexingPipeline

if TYPE_CHECKING:
    from oce.application.queue import Queue
    from oce.application.uow import UnitOfWorkFactory
    from oce.domain.chunk import Chunker
    from oce.domain.services.embedder import Embedder
    from oce.domain.services.path_search import PathSearchStore
    from oce.domain.services.search import VectorIndex


class EmbedWorker:
    """异步嵌入 worker（持有 queue + uow_factory + pipeline 依赖）"""

    def __init__(
        self,
        *,
        queue: Queue,
        uow_factory: UnitOfWorkFactory,
        chunker: Chunker,
        embedder: Embedder,
        vector_index: VectorIndex,
        path_store: PathSearchStore | None = None,
        concurrency: int = 2,
        max_retries: int = 3,
    ) -> None:
        self._queue = queue
        self._uow_factory = uow_factory
        self._chunker = chunker
        self._embedder = embedder
        self._vector_index = vector_index
        self._path_store = path_store
        self._concurrency = max(1, concurrency)
        self._max_retries = max_retries
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """启动前先恢复上次崩溃残留，再拉起 N 个消费协程"""
        if self._running:
            return
        self._running = True
        if hasattr(self._queue, "recover_processing"):
            recovered = await self._queue.recover_processing()
            if recovered:
                logger.info("EmbedWorker: 恢复 {} 条处理中残留任务", recovered)
        self._tasks = [
            asyncio.create_task(self._loop(i)) for i in range(self._concurrency)
        ]
        logger.info("EmbedWorker 启动，{} 个消费协程", self._concurrency)

    async def stop(self) -> None:
        """停止：置标志 + 取消协程并等待退出"""
        self._running = False
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        logger.info("EmbedWorker 已停止")

    def _build_pipeline(self, uow) -> IndexingPipeline:
        """每条消息一个 pipeline，repo 绑定当前 UoW，避免协程间竞态"""
        return IndexingPipeline(
            chunker=self._chunker,
            embedder=self._embedder,
            vector_index=self._vector_index,
            blob_repo=uow.blobs,
            chunk_repo=uow.chunks,
            path_store=self._path_store,
        )

    async def _loop(self, worker_id: int) -> None:
        """单个消费协程：取任务 → 嵌入 → ack/fail"""
        while self._running:
            try:
                blob_name = await self._queue.dequeue(timeout=5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("worker#{} dequeue 异常: {}", worker_id, e)
                await asyncio.sleep(1)
                continue

            if blob_name is None:
                await asyncio.sleep(0.05)
                continue

            try:
                # embed_pending 内部会从 staging 取原文切块(如需),然后嵌入、删 staging
                async with self._uow_factory() as uow:
                    pipeline = self._build_pipeline(uow)
                    n = await pipeline.embed_pending(
                        [blob_name],
                        mark_failures=False,
                    )
                    await uow.commit()

                await self._queue.ack(blob_name)
                logger.debug("worker#{} processed blob {} ({} chunks embed)", worker_id, blob_name[:12], n)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("worker#{} process 失败 blob {}: {}", worker_id, blob_name[:12], e)
                try:
                    await self._queue.fail(blob_name)
                    should_retry = False
                    # DB 层处理重试:超限则 mark_error + 删 staging,未超限保留 staging 供重试
                    async with self._uow_factory() as uow:
                        blob = await uow.blobs.get(blob_name)
                        if blob:
                            exceeded = blob.increment_retry(self._max_retries)
                            if exceeded:
                                blob.mark_error(str(e))
                            await uow.blobs.save(blob)
                            if exceeded:
                                # 超限放弃,清理 staging
                                await uow.blobs.delete_staging(blob_name)
                                logger.error("worker#{} blob {} 重试超限 → error, staging 已清理", worker_id, blob_name[:12])
                            else:
                                should_retry = True
                                logger.info("worker#{} blob {} retry_count={}, staging 保留供重试", worker_id, blob_name[:12], blob.retry_count)
                            await uow.commit()
                    if should_retry:
                        await self._queue.enqueue(blob_name)
                except Exception as e2:
                    logger.error("worker#{} fail 处理异常: {}", worker_id, e2)
