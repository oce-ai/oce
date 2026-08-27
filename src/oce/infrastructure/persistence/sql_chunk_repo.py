"""SQLAlchemy chunk 元数据仓储。向量只写入 Milvus。"""

from __future__ import annotations

from typing import Sequence

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from oce.domain.chunk import Chunk, LocatedChunk
from oce.infrastructure.persistence.models import BlobChunkModel, BlobModel, ChunkModel
from oce.domain.repositories import ChunkRepository


class SqlChunkRepository(ChunkRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _insert(self):
        bind = self.session.get_bind()
        return sqlite_insert if bind.dialect.name == "sqlite" else pg_insert

    async def get(self, content_hash: str) -> Chunk | None:
        result = await self.session.execute(
            select(ChunkModel).where(ChunkModel.content_hash == content_hash)
        )
        row = result.scalar_one_or_none()
        return self._row_to_domain(row) if row else None

    async def get_many(self, content_hashes: Sequence[str]) -> dict[str, Chunk]:
        if not content_hashes:
            return {}
        result = await self.session.execute(
            select(ChunkModel).where(ChunkModel.content_hash.in_(content_hashes))
        )
        return {row.content_hash: self._row_to_domain(row) for row in result.scalars()}

    async def save(self, chunk: Chunk) -> None:
        await self.save_many([chunk])

    async def save_many(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        values = [
            {
                "content_hash": chunk.content_hash,
                "content": chunk.content,
                "content_size": len(chunk.content.encode("utf-8")),
                "chunk_type": chunk.chunk_type,
                "embedded": False,  # 新切的 chunk 默认未嵌入
            }
            for chunk in chunks
        ]
        stmt = self._insert()(ChunkModel).values(values)
        stmt = stmt.on_conflict_do_nothing(index_elements=["content_hash"])
        await self.session.execute(stmt)

    async def mark_embedded(self, content_hashes: Sequence[str]) -> None:
        """标记 chunks 已嵌入到 Milvus"""
        from oce.infrastructure.persistence.models import ChunkModel
        from sqlalchemy import update

        if not content_hashes:
            return
        stmt = (
            update(ChunkModel)
            .where(ChunkModel.content_hash.in_(content_hashes))
            .values(embedded=True)
        )
        await self.session.execute(stmt)

    async def find_pending_for_blobs(
        self,
        blob_names: Sequence[str],
        limit: int | None = None,
    ) -> list[LocatedChunk]:
        if not blob_names:
            return []

        stmt = (
            select(
                BlobChunkModel.blob_name,
                ChunkModel.content_hash,
                BlobModel.path,
                ChunkModel.content,
                BlobChunkModel.start_line,
                BlobChunkModel.end_line,
            )
            .join(ChunkModel, ChunkModel.content_hash == BlobChunkModel.content_hash)
            .join(BlobModel, BlobModel.blob_name == BlobChunkModel.blob_name)
            .where(
                BlobChunkModel.blob_name.in_(blob_names),
                BlobModel.status == "pending",
            )
            .order_by(BlobChunkModel.blob_name, BlobChunkModel.chunk_index)
        )
        if limit is not None:
            stmt = stmt.limit(limit)
        rows = (await self.session.execute(stmt)).all()
        return [
            LocatedChunk(
                blob_name=row.blob_name,
                content_hash=row.content_hash,
                path=row.path,
                content=row.content,
                start_line=row.start_line,
                end_line=row.end_line,
            )
            for row in rows
        ]

    @staticmethod
    def _row_to_domain(row: ChunkModel) -> Chunk:
        line_count = max(1, len(row.content.splitlines()))
        return Chunk(
            content_hash=row.content_hash,
            path="",
            content=row.content,
            start_line=1,
            end_line=line_count,
            chunk_type=row.chunk_type,
        )
