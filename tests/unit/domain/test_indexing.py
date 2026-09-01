"""IndexingPipeline 领域服务测试

用内存 Fake repo 验证：切块入库、懒嵌入、状态转换、事件发布。
"""

from __future__ import annotations

import hashlib

import pytest

from oce.domain.blob.blob import BlobStatus
from oce.domain.chunk import Chunk, RecursiveChunker, LocatedChunk
from oce.domain.services.indexing import (
    EVENT_BLOB_CREATED,
    EVENT_BLOB_FAILED,
    EVENT_BLOB_READY,
    IndexingPipeline,
)
from oce.shared.events import DomainEvent, EventBus


class FakeBlobRepo:
    """BlobRepository 内存替身（仅实现 pipeline 用到的方法）"""

    def __init__(self):
        self.blobs: dict[str, object] = {}
        self.staging: dict[str, bytes] = {}

    async def get(self, blob_name: str):
        return self.blobs.get(blob_name)

    async def save(self, blob) -> None:
        self.blobs[blob.blob_name] = blob

    async def get_staging(self, blob_name: str) -> str | None:
        return self.staging.get(blob_name)

    async def save_staging(self, blob_name: str, content: str) -> None:
        # blob_staging.content 是 Text 列；这里跟着断言类型，防止再退回 bytes。
        assert isinstance(content, str), "staging 原文必须是 str"
        self.staging[blob_name] = content

    async def delete_staging(self, blob_name: str) -> None:
        self.staging.pop(blob_name, None)

    async def find_pending(self, blob_names=None) -> list:
        names = set(blob_names) if blob_names is not None else None
        return [
            b for b in self.blobs.values()
            if b.status == BlobStatus.PENDING
            and (names is None or b.blob_name in names)
        ]


class FakeChunkRepo:
    """ChunkRepository 内存替身"""

    def __init__(self):
        self.chunks: dict[str, object] = {}
        self.pending: list[LocatedChunk] = []
        self.last_blob_name: str = ""  # 记录最后操作的 blob_name

    async def save_many(self, chunks) -> None:
        for c in chunks:
            self.chunks[c.content_hash] = c
            # 同时添加到 pending 列表，模拟待嵌入状态
            # blob_name 从路径派生（简化）
            blob_name = hashlib.sha256(c.path.encode()).hexdigest()
            located = LocatedChunk(
                blob_name=blob_name,
                content_hash=c.content_hash,
                path=c.path,
                content=c.content,
                start_line=c.start_line,
                end_line=c.end_line,
            )
            self.pending.append(located)

    async def find_pending_for_blobs(self, blob_names, limit=None) -> list[LocatedChunk]:
        # 返回所有 pending 块（测试简化，实际应该按 blob_names 过滤）
        result = list(self.pending)
        if limit:
            result = result[:limit]
        return result

    async def mark_embedded(self, content_hashes: list[str]) -> None:
        # 标记为已嵌入，从 pending 移除
        self.pending = [c for c in self.pending if c.content_hash not in content_hashes]


class FakeVectorIndex:
    def __init__(self):
        self.items: list[dict] = []

    async def upsert(self, items: list[dict]) -> None:
        self.items.extend(items)

    async def delete(self, blob_names: list[str]) -> None:
        return None


class FakeEmbedder:
    """确定性假 embedder"""

    def __init__(self):
        self.calls: list[list[str]] = []

    async def embed_documents(self, texts):
        self.calls.append(texts)
        return [[1.0] * 4 for _ in texts]


def _blob_name(path: str, content: str) -> str:
    return hashlib.sha256((path + content).encode("utf-8")).hexdigest()


@pytest.fixture
def indexing_pipeline(monkeypatch):
    # 启用嵌入，确保 embed_pending 返回正确的数量
    monkeypatch.setenv("EMBED_ENABLED", "true")

    blob_repo = FakeBlobRepo()
    chunk_repo = FakeChunkRepo()
    embedder = FakeEmbedder()
    event_bus = EventBus()
    events: list[DomainEvent] = []

    async def _collect(event: DomainEvent) -> None:
        events.append(event)

    event_bus.subscribe(EVENT_BLOB_CREATED, _collect)
    event_bus.subscribe(EVENT_BLOB_FAILED, _collect)
    event_bus.subscribe(EVENT_BLOB_READY, _collect)

    pipe = IndexingPipeline(
        chunker=RecursiveChunker(chunk_size=6000, chunk_overlap=200),
        embedder=embedder,
        vector_index=FakeVectorIndex(),
        blob_repo=blob_repo,
        chunk_repo=chunk_repo,
        event_bus=event_bus,
    )
    pipe._events = events  # type: ignore[attr-defined]
    return pipe


class TestIngest:
    async def test_ingest_saves_blob_and_chunks(self, indexing_pipeline):
        # 100 行每行约 10 字符 = 1000 字符，需要调整 chunker 以产生多个块
        content = "\n".join(f"line{i}" for i in range(100))
        name = _blob_name("src/a.py", content)

        # 设置小块 chunker 以测试分块行为
        original_chunker = indexing_pipeline.chunker
        indexing_pipeline.chunker = RecursiveChunker(chunk_size=400, chunk_overlap=50)

        count = await indexing_pipeline.ingest(name, "src/a.py", content)

        # 异步模式：ingest 返回 0，实际切块在 embed_pending
        assert count == 0

        # 执行切块
        embedded_count = await indexing_pipeline.embed_pending([name])
        assert embedded_count >= 2  # 1000 字符 / 400 字符

        blob = indexing_pipeline.blob_repo.blobs[name]
        assert blob.status == BlobStatus.READY
        assert len(blob.chunks) == embedded_count

        # 恢复原 chunker
        indexing_pipeline.chunker = original_chunker

    async def test_ingest_empty_content_keeps_blob(self, indexing_pipeline):
        name = _blob_name("src/empty.py", "")
        count = await indexing_pipeline.ingest(name, "src/empty.py", "")

        assert count == 0

        # 异步模式：空文件直接进入 pending，等待 embed_pending 处理
        blob = indexing_pipeline.blob_repo.blobs[name]
        assert blob.status == BlobStatus.PENDING

        # 执行 embed_pending，空内容应该被标记为 ready
        await indexing_pipeline.embed_pending([name])
        blob = indexing_pipeline.blob_repo.blobs[name]
        assert blob.status == BlobStatus.READY
        assert blob.chunks == []

    async def test_ingest_ignored_source_never_reaches_chunker(self, indexing_pipeline):
        class FailingChunker:
            def chunk(self, _content, _path):
                raise AssertionError("ignored source reached chunker")

        indexing_pipeline.chunker = FailingChunker()
        content = '<svg><path d="M0 0" /></svg>'
        name = _blob_name("assets/logo.svg", content)

        count = await indexing_pipeline.ingest(name, "assets/logo.svg", content)

        assert count == 0
        blob = indexing_pipeline.blob_repo.blobs[name]
        assert blob.status == BlobStatus.READY
        assert blob.file_type == "ignored"

    async def test_ingest_binary_source_never_reaches_chunker(self, indexing_pipeline):
        class FailingChunker:
            def chunk(self, _content, _path):
                raise AssertionError("binary source reached chunker")

        indexing_pipeline.chunker = FailingChunker()
        content = "PNG\x00data"
        name = _blob_name("assets/logo.dat", content)

        count = await indexing_pipeline.ingest(name, "assets/logo.dat", content)

        assert count == 0
        assert indexing_pipeline.blob_repo.blobs[name].file_type == "binary"

    async def test_ingest_publishes_created_event(self, indexing_pipeline):
        content = "print('hi')\n"
        name = _blob_name("src/hi.py", content)
        await indexing_pipeline.ingest(name, "src/hi.py", content)

        types = [e.event_type for e in indexing_pipeline._events]
        assert EVENT_BLOB_CREATED in types
        assert any(e.data.get("blob_name") == name for e in indexing_pipeline._events)

    async def test_ingest_same_blob_is_idempotent(self, indexing_pipeline):
        content = "print('same')\n"
        name = _blob_name("src/same.py", content)

        first = await indexing_pipeline.ingest(name, "src/same.py", content)
        second = await indexing_pipeline.ingest(name, "src/same.py", content)

        assert first == second
        created = [
            event
            for event in indexing_pipeline._events
            if event.event_type == EVENT_BLOB_CREATED
        ]
        assert len(created) == 1


class TestEmbedPending:
    async def test_embed_pending_marks_blob_ready(self, indexing_pipeline):
        # 完全重置 fixture 状态，避免其他测试的干扰
        indexing_pipeline.chunk_repo.pending.clear()
        indexing_pipeline.chunk_repo.chunks.clear()
        indexing_pipeline.vector_index.items.clear()
        indexing_pipeline.blob_repo.blobs.clear()
        indexing_pipeline.blob_repo.staging.clear()

        content = "print('hello')\n"
        name = _blob_name("src/hello.py", content)

        # 模拟「已入库但未嵌入」的 chunk
        chunk = Chunk(
            content_hash=Chunk.compute_hash(content),
            path="src/hello.py",
            content=content,
            start_line=1,
            end_line=1,
        )

        # 创建 blob（pending 状态，已有 chunks）
        from oce.domain.blob.blob import Blob, BlobStatus
        blob = Blob(
            blob_name=name,
            path="src/hello.py",
            status=BlobStatus.PENDING,
            chunks=[chunk.to_ref()],  # 已经切块，只是未嵌入
            content_size=len(content),
            language="python",
            file_type="text",
        )
        await indexing_pipeline.blob_repo.save(blob)

        # 设置 pending 列表，模拟待嵌入状态
        indexing_pipeline.chunk_repo.pending = [
            LocatedChunk(name, chunk.content_hash, chunk.path, chunk.content, 1, 1)
        ]

        embedded = await indexing_pipeline.embed_pending([name])

        assert embedded == 1
        blob = indexing_pipeline.blob_repo.blobs[name]
        assert blob.status == BlobStatus.READY
        # 验证嵌入的 chunk
        assert len(indexing_pipeline.vector_index.items) == 1
        assert indexing_pipeline.vector_index.items[0]["content_hash"] == chunk.content_hash

    async def test_embed_pending_disabled_keeps_pending_and_staging(
        self, indexing_pipeline, monkeypatch
    ):
        """回归：EMBED_ENABLED=false 时，有 chunk 的 blob 必须停在 pending 且保留
        staging，绝不 mark_ready。

        复现线上静默失效：切块已落库、Milvus 零向量，若此时点亮 READY，检索恒空，
        且内容寻址幂等会让客户端重传也无法自愈。修复后 blob 保持 pending、原文保留，
        待开关恢复重新入队即可无损补嵌。
        """
        from oce.domain.blob.blob import Blob, BlobStatus
        from oce.shared.config import get_settings

        indexing_pipeline.chunk_repo.pending.clear()
        indexing_pipeline.chunk_repo.chunks.clear()
        indexing_pipeline.vector_index.items.clear()
        indexing_pipeline.blob_repo.blobs.clear()
        indexing_pipeline.blob_repo.staging.clear()

        content = "print('hello')\n"
        name = _blob_name("src/hello.py", content)
        chunk = Chunk(
            content_hash=Chunk.compute_hash(content),
            path="src/hello.py",
            content=content,
            start_line=1,
            end_line=1,
        )
        blob = Blob(
            blob_name=name,
            path="src/hello.py",
            status=BlobStatus.PENDING,
            chunks=[chunk.to_ref()],  # 已切块，只是尚未嵌入
            content_size=len(content),
            language="python",
            file_type="text",
        )
        await indexing_pipeline.blob_repo.save(blob)
        await indexing_pipeline.blob_repo.save_staging(name, content)
        indexing_pipeline.chunk_repo.pending = [
            LocatedChunk(name, chunk.content_hash, chunk.path, chunk.content, 1, 1)
        ]

        # 关掉嵌入开关；get_settings 有 lru_cache，必须 cache_clear 才能让新值穿透。
        monkeypatch.setenv("EMBED_ENABLED", "false")
        get_settings.cache_clear()
        try:
            embedded = await indexing_pipeline.embed_pending([name])
        finally:
            get_settings.cache_clear()  # 避免 false 泄漏到后续用例

        assert embedded == 0
        assert indexing_pipeline.vector_index.items == []  # 未写任何向量
        blob = indexing_pipeline.blob_repo.blobs[name]
        assert blob.status == BlobStatus.PENDING            # 核心：不再假 READY
        assert name in indexing_pipeline.blob_repo.staging  # 原文保留，供恢复后补嵌
        assert len(indexing_pipeline.chunk_repo.pending) == 1  # chunk 未被消费
        assert EVENT_BLOB_READY not in [
            e.event_type for e in indexing_pipeline._events
        ]

    async def test_embed_pending_no_pending_returns_zero(self, indexing_pipeline):
        name = _blob_name("src/x.py", "print(1)\n")
        await indexing_pipeline.ingest(name, "src/x.py", "print(1)\n")

        # 清空 pending 并标记 blob 为 ready，模拟已完成嵌入的状态
        indexing_pipeline.chunk_repo.pending.clear()
        blob = indexing_pipeline.blob_repo.blobs[name]
        blob.mark_ready()
        await indexing_pipeline.blob_repo.save(blob)

        assert await indexing_pipeline.embed_pending([name]) == 0

    async def test_embed_failure_marks_blob_error(self, indexing_pipeline):
        class FailingEmbedder:
            async def embed_documents(self, _texts):
                raise RuntimeError("provider rejected input")

        content = "print('broken')\n"
        name = _blob_name("src/broken.py", content)
        await indexing_pipeline.ingest(name, "src/broken.py", content)
        chunk = Chunk(
            content_hash=Chunk.compute_hash(content),
            path="src/broken.py",
            content=content,
            start_line=1,
            end_line=1,
        )
        indexing_pipeline.chunk_repo.pending = [
            LocatedChunk(name, chunk.content_hash, chunk.path, chunk.content, 1, 1)
        ]
        indexing_pipeline.embedder = FailingEmbedder()

        with pytest.raises(RuntimeError, match="provider rejected input"):
            await indexing_pipeline.embed_pending([name])

        blob = indexing_pipeline.blob_repo.blobs[name]
        assert blob.status == BlobStatus.ERROR
        assert blob.error_message == "provider rejected input"
        assert EVENT_BLOB_FAILED in [event.event_type for event in indexing_pipeline._events]
