"""IndexingPipeline 领域服务 - 索引编排

职责（对应旧三表生产架构的领域抽象）：
- ingest:        切块 → Blob/Chunk 入库（embedding 留空，懒嵌入）
- embed_pending: 待嵌入 chunk → 向量化 → 写回 → Blob 置 ready

事件（经 EventBus 发布，event_type 常量见下）：
- blob.created / blob.ready / blob.failed
"""

from __future__ import annotations

from typing import Sequence

from loguru import logger

from oce.domain.blob.blob import Blob, BlobStatus
from oce.domain.chunk import Chunker, is_meaningful
from oce.domain.chunk.lang import detect_language
from oce.domain.services.embedder import Embedder
from oce.domain.services.path_document_builder import (
    build_path_document,
    is_indexable_path,
)
from oce.domain.services.path_search import PathSearchStore
from oce.domain.services.search import VectorIndex
from oce.domain.services.source_filter import is_binary_source, is_ignored_source_path
from oce.shared.events import DomainEvent, EventBus
from oce.domain.repositories import BlobRepository, ChunkRepository

EVENT_BLOB_CREATED = "blob.created"
EVENT_BLOB_READY = "blob.ready"
EVENT_BLOB_FAILED = "blob.failed"


class IndexingPipeline:
    """索引管道：切块入库 + 懒嵌入"""

    def __init__(
        self,
        *,
        chunker: Chunker,
        embedder: Embedder,
        vector_index: VectorIndex,
        blob_repo: BlobRepository,
        chunk_repo: ChunkRepository,
        event_bus: EventBus | None = None,
        embed_batch_size: int = 64,
        path_store: PathSearchStore | None = None,
    ) -> None:
        if embed_batch_size < 1:
            raise ValueError("embed_batch_size must be positive")
        self.chunker = chunker
        self.embedder = embedder
        self.vector_index = vector_index
        self.blob_repo = blob_repo
        self.chunk_repo = chunk_repo
        self.event_bus = event_bus
        self.embed_batch_size = embed_batch_size
        self.path_store = path_store

    async def ingest(self, blob_name: str, path: str, content: str) -> int:
        """轻量入库:只写元数据,切块推给 worker。立刻返回 0。

        高吞吐设计:upload 接口只做轻量 IO,chunking+embedding 全部异步。
        blob_name 由调用方按内容哈希算好传入。
        客户端用 find_missing 轮询 ready 状态。
        """
        existing = await self.blob_repo.get(blob_name)
        is_binary = is_binary_source(content)
        if is_binary or is_ignored_source_path(path):
            if existing is not None and existing.chunks:
                await self.vector_index.delete([blob_name])
                await self.blob_repo.delete(blob_name)
            blob = Blob(
                blob_name=blob_name,
                path=path,
                status=BlobStatus.READY,
                content_size=len(content.encode("utf-8")),
                language=detect_language(path),
                file_type="binary" if is_binary else "ignored",
            )
            await self.blob_repo.save(blob)
            return 0

        if existing is not None and existing.status in (
            BlobStatus.PENDING,
            BlobStatus.READY,
        ):
            existing.touch()
            await self.blob_repo.save(existing)
            return 0  # 异步模式:不返回 chunk_count,客户端轮询

        # 只写元数据,不切块
        blob = Blob(
            blob_name=blob_name,
            path=path,
            status=BlobStatus.PENDING,
            chunks=[],  # 空,worker 负责切块后回填
            content_size=len(content.encode("utf-8")),
            language=detect_language(path),
            file_type="text",
        )
        await self.blob_repo.save(blob)

        # 保存原文到 staging，供后续 embed_pending 切块使用。
        # blob_staging.content 是 Text 列，编码成 bytes 会被驱动拒绝。
        await self.blob_repo.save_staging(blob_name, content)

        if self.event_bus is not None:
            await self.event_bus.publish(DomainEvent(
                event_type=EVENT_BLOB_CREATED,
                data={"blob_name": blob_name, "path": path, "chunk_count": 0},
            ))
        return 0

    async def embed_pending(
        self,
        blob_names: Sequence[str] | None = None,
        *,
        mark_failures: bool = True,
    ) -> int:
        """处理待嵌入 blob:切块(如需)→ 向量化 → 写回 → 置 ready。

        返回嵌入条数。只处理 pending 状态的 Blob。
        如果 blob.chunks 为空,说明 ingest 只写了元数据,需要先从 staging 取原文切块。

        如果嵌入开关关闭(EMBED_ENABLED=false),则只完成切块、不嵌入,blob 直接置 ready。
        """
        from oce.shared.config import get_settings

        blobs = await self.blob_repo.find_pending(blob_names)
        if not blobs:
            return 0

        # 收集本次调用中置为 ready 的 blob，统一补写路径索引
        ready_blobs: list[Blob] = []

        # 第一阶段:补切块(针对 ingest 只写元数据的 blob)
        for blob in blobs:
            if not blob.chunks:
                # staging 里有原文,需要切块
                content = await self.blob_repo.get_staging(blob.blob_name)
                if content is None:
                    if blob.content_size == 0:
                        # 空文件没有 staging 内容（或为空串），无需切块，直接 ready
                        blob.mark_ready()
                        await self.blob_repo.save(blob)
                        ready_blobs.append(blob)
                        continue
                    # staging 不存在,可能被清理或异常,跳过
                    blob.mark_error("staging content not found")
                    await self.blob_repo.save(blob)
                    continue

                # RecursiveChunker 已经过滤了无意义的块，无需再次过滤
                chunks = list(self.chunker.chunk(content, blob.path))
                if chunks:
                    # 保存 chunks（包含 chunk_type）
                    await self.chunk_repo.save_many(chunks)
                    blob.chunks = [c.to_ref() for c in chunks]
                    await self.blob_repo.save(blob)
                else:
                    # 无有效内容,直接 ready
                    blob.mark_ready()
                    await self.blob_repo.save(blob)
                    await self.blob_repo.delete_staging(blob.blob_name)
                    ready_blobs.append(blob)
                    continue

        # 检查嵌入开关
        settings = get_settings()
        if not settings.embedding.enabled:
            # 嵌入关闭:只完成切块,直接 ready,不写 Milvus。
            # 嵌入关闭时 embedder 通常不可用，路径向量无法生成，跳过路径索引写入。
            for blob in blobs:
                if blob.status == BlobStatus.PENDING:
                    blob.mark_ready()
                    await self.blob_repo.save(blob)
                    await self.blob_repo.delete_staging(blob.blob_name)
            return 0

        # 第二阶段:嵌入
        pending = await self.chunk_repo.find_pending_for_blobs(
            [blob.blob_name for blob in blobs]
        )
        embedded = 0
        try:
            for offset in range(0, len(pending), self.embed_batch_size):
                chunk_batch = pending[offset:offset + self.embed_batch_size]

                vectors = await self.embedder.embed_documents(
                    [chunk.embedding_text() for chunk in chunk_batch]
                )
                if len(vectors) != len(chunk_batch):
                    raise RuntimeError(
                        "Embedding count mismatch: "
                        f"expected {len(chunk_batch)}, got {len(vectors)}"
                    )
                await self.vector_index.upsert([
                    {
                        "chunk_id": chunk.chunk_id,
                        "content_hash": chunk.content_hash,
                        "blob_name": chunk.blob_name,
                        "content": chunk.content,
                        "vector": vector,
                        "metadata": {
                            "path": chunk.path,
                            "start_line": chunk.start_line,
                            "end_line": chunk.end_line,
                        },
                    }
                    for chunk, vector in zip(chunk_batch, vectors)
                ])
                # 标记已嵌入
                await self.chunk_repo.mark_embedded([c.content_hash for c in chunk_batch])
                embedded += len(vectors)
        except Exception as exc:
            if mark_failures:
                for blob in blobs:
                    blob.mark_error(str(exc))
                    await self.blob_repo.save(blob)
                    if self.event_bus is not None:
                        await self.event_bus.publish(DomainEvent(
                            event_type=EVENT_BLOB_FAILED,
                            data={
                                "blob_name": blob.blob_name,
                                "path": blob.path,
                                "error": str(exc),
                            },
                        ))
            raise

        # 第三阶段:标记 ready + 清理 staging
        for blob in blobs:
            if blob.status == BlobStatus.PENDING:  # 跳过第一阶段标记 error 的
                blob.mark_ready()
                await self.blob_repo.save(blob)
                await self.blob_repo.delete_staging(blob.blob_name)
                ready_blobs.append(blob)
                if self.event_bus is not None:
                    await self.event_bus.publish(DomainEvent(
                        event_type=EVENT_BLOB_READY,
                        data={"blob_name": blob.blob_name, "path": blob.path},
                    ))
        # 路径索引写入失败不应影响主索引（chunk 已嵌入、blob 已 ready），仅记日志
        await self._index_paths(ready_blobs)
        return embedded

    async def _index_paths(self, blobs: Sequence[Blob]) -> None:
        """把 ready blob 的路径写入路径索引（文件名查询专用通道）。

        路径索引只依赖路径文本与扩展名语义，不需要 chunk 内容，因此放在
        embed_pending 完成后统一批量写入，避免 ingest 阶段多一次 embedding
        拖慢上传。依赖/构建/二进制路径由 is_indexable_path 排除。
        """
        if self.path_store is None:
            return
        indexable = [blob for blob in blobs if is_indexable_path(blob.path)]
        if not indexable:
            return
        docs = [
            {
                "blob_name": blob.blob_name,
                "path": blob.path,
                "path_document": build_path_document(blob.path),
            }
            for blob in indexable
        ]
        try:
            vectors = await self.embedder.embed_documents(
                [doc["path_document"] for doc in docs]
            )
            for doc, vector in zip(docs, vectors):
                doc["path_id"] = f"path_{doc['blob_name']}"
                doc["path_vector"] = vector
            result = await self.path_store.insert(docs)
            logger.info(
                "path index write: {} blobs ({})",
                len(docs),
                result.get("inserted", 0),
            )
        except Exception as exc:
            logger.warning("path index write failed for {} blobs: {}", len(docs), exc)
