"""SQLAlchemy Blob 聚合仓储。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from oce.domain.blob.blob import Blob, BlobStatus
from oce.domain.chunk import ChunkRef
from oce.infrastructure.persistence.models import (
    BlobChunkModel,
    BlobModel,
    ChunkModel,
    SymbolOccurrenceModel,
)
from oce.domain.repositories import BlobRepository
from oce.infrastructure.persistence.symbol_extractor import SymbolExtractor


class SqlBlobRepository(BlobRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session
        self._symbol_extractor = SymbolExtractor()

    def _insert(self):
        bind = self.session.get_bind()
        return sqlite_insert if bind.dialect.name == "sqlite" else pg_insert

    async def get(self, blob_name: str) -> Blob | None:
        row = (
            await self.session.execute(
                select(BlobModel).where(BlobModel.blob_name == blob_name)
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return self._row_to_domain(row, await self._load_chunks(blob_name))

    async def get_many(self, blob_names: Sequence[str]) -> dict[str, Blob]:
        if not blob_names:
            return {}
        rows = (
            await self.session.execute(
                select(BlobModel).where(BlobModel.blob_name.in_(blob_names))
            )
        ).scalars().all()
        chunks = await self._load_chunks_many(blob_names)
        return {
            row.blob_name: self._row_to_domain(row, chunks.get(row.blob_name, []))
            for row in rows
        }

    async def exists(self, blob_name: str) -> bool:
        count = await self.session.scalar(
            select(func.count()).select_from(BlobModel).where(BlobModel.blob_name == blob_name)
        )
        return bool(count)

    async def exists_many(self, blob_names: Sequence[str]) -> dict[str, bool]:
        if not blob_names:
            return {}
        rows = await self.session.execute(
            select(BlobModel.blob_name).where(BlobModel.blob_name.in_(blob_names))
        )
        existing = set(rows.scalars())
        return {name: name in existing for name in blob_names}

    async def save(self, blob: Blob) -> None:
        await self.save_many([blob])

    async def save_many(self, blobs: Sequence[Blob]) -> None:
        if not blobs:
            return
        values = [
            {
                "blob_name": blob.blob_name,
                "path": blob.path,
                "content_size": blob.content_size,
                "language": blob.language,
                "file_type": blob.file_type,
                "status": blob.status.value,
                "retry_count": blob.retry_count,
                "last_seen": blob.last_seen,
                "created_at": blob.created_at,
                "error_message": blob.error_message,
            }
            for blob in blobs
        ]
        stmt = self._insert()(BlobModel).values(values)
        stmt = stmt.on_conflict_do_update(
            index_elements=["blob_name"],
            set_={
                "path": stmt.excluded.path,
                "content_size": stmt.excluded.content_size,
                "language": stmt.excluded.language,
                "file_type": stmt.excluded.file_type,
                "status": stmt.excluded.status,
                "retry_count": stmt.excluded.retry_count,
                "last_seen": stmt.excluded.last_seen,
                "error_message": stmt.excluded.error_message,
            },
        )
        await self.session.execute(stmt)
        for blob in blobs:
            await self._save_blob_chunks(blob.blob_name, blob.chunks)
            await self._extract_and_save_symbols(blob)

    async def delete(self, blob_name: str) -> None:
        await self.delete_many([blob_name])

    async def delete_many(self, blob_names: Sequence[str]) -> None:
        if not blob_names:
            return
        content_hashes = list(
            (
                await self.session.execute(
                    select(BlobChunkModel.content_hash)
                    .where(BlobChunkModel.blob_name.in_(blob_names))
                    .distinct()
                )
            ).scalars()
        )
        await self.session.execute(
            delete(BlobChunkModel).where(BlobChunkModel.blob_name.in_(blob_names))
        )
        await self.session.execute(delete(BlobModel).where(BlobModel.blob_name.in_(blob_names)))
        if content_hashes:
            referenced = select(BlobChunkModel.content_hash).where(
                BlobChunkModel.content_hash == ChunkModel.content_hash
            )
            await self.session.execute(
                delete(ChunkModel).where(
                    ChunkModel.content_hash.in_(content_hashes),
                    ~referenced.exists(),
                )
            )

    async def find_pending(self, blob_names: Sequence[str] | None = None) -> list[Blob]:
        stmt = select(BlobModel).where(BlobModel.status == BlobStatus.PENDING.value)
        if blob_names is not None:
            if not blob_names:
                return []
            stmt = stmt.where(BlobModel.blob_name.in_(blob_names))
        rows = (await self.session.execute(stmt)).scalars().all()
        chunks = await self._load_chunks_many([row.blob_name for row in rows])
        return [self._row_to_domain(row, chunks.get(row.blob_name, [])) for row in rows]

    async def find_expired(self, ttl_days: int, batch_size: int = 1000) -> list[str]:
        threshold = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        rows = await self.session.execute(
            select(BlobModel.blob_name)
            .where(BlobModel.last_seen < threshold)
            .limit(batch_size)
        )
        return list(rows.scalars())

    async def touch_last_seen(self, blob_names: Sequence[str]) -> None:
        if blob_names:
            await self.session.execute(
                update(BlobModel)
                .where(BlobModel.blob_name.in_(blob_names))
                .values(last_seen=datetime.now(timezone.utc))
            )

    # ── blob_staging 操作 ────────────────────────────────────────────────

    async def get_staging(self, blob_name: str) -> str | None:
        """读取 staging 原文，不存在返回 None"""
        from oce.infrastructure.persistence.models import BlobStagingModel
        result = await self.session.execute(
            select(BlobStagingModel.content).where(BlobStagingModel.blob_name == blob_name)
        )
        row = result.scalar_one_or_none()
        # 空文件保存的是空串，不能把空串当成“不存在”（否则空文件会被误判为 staging 丢失）
        return row if row is not None else None

    async def save_staging(self, blob_name: str, content: str) -> None:
        """保存 staging 原文，已存在则跳过（UPSERT 幂等）"""
        from oce.infrastructure.persistence.models import BlobStagingModel
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        # 根据 dialect 选择 insert 语句
        if self.session.bind.dialect.name == "postgresql":
            stmt = pg_insert(BlobStagingModel).values(
                blob_name=blob_name, content=content
            ).on_conflict_do_nothing(index_elements=["blob_name"])
        else:  # SQLite
            stmt = sqlite_insert(BlobStagingModel).values(
                blob_name=blob_name, content=content
            ).on_conflict_do_nothing()

        await self.session.execute(stmt)

    async def delete_staging(self, blob_name: str) -> None:
        """删除 staging 原文（worker 消费完后调用）"""
        from oce.infrastructure.persistence.models import BlobStagingModel
        await self.session.execute(
            delete(BlobStagingModel).where(BlobStagingModel.blob_name == blob_name)
        )

    async def _load_chunks(self, blob_name: str) -> list[ChunkRef]:
        return (await self._load_chunks_many([blob_name])).get(blob_name, [])

    async def _load_chunks_many(self, blob_names: Sequence[str]) -> dict[str, list[ChunkRef]]:
        if not blob_names:
            return {}
        rows = (
            await self.session.execute(
                select(BlobChunkModel)
                .where(BlobChunkModel.blob_name.in_(blob_names))
                .order_by(BlobChunkModel.blob_name, BlobChunkModel.chunk_index)
            )
        ).scalars()
        result: dict[str, list[ChunkRef]] = {}
        for row in rows:
            result.setdefault(row.blob_name, []).append(
                ChunkRef(row.content_hash, row.start_line, row.end_line)
            )
        return result

    async def _save_blob_chunks(self, blob_name: str, chunks: Sequence[ChunkRef]) -> None:
        if not chunks:
            return
        values = [
            {
                "blob_name": blob_name,
                "content_hash": chunk.content_hash,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_index": index,
            }
            for index, chunk in enumerate(chunks)
        ]
        stmt = self._insert()(BlobChunkModel).values(values)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["blob_name", "content_hash", "start_line", "end_line"]
        )
        await self.session.execute(stmt)

    async def _extract_and_save_symbols(self, blob: Blob) -> None:
        """从 blob 的所有 chunks 提取标识符并写入 symbol_occurrences 表。"""
        import logging
        logger = logging.getLogger(__name__)

        if not blob.chunks:
            logger.debug(f"Blob {blob.blob_name}: no chunks, skipping symbol extraction")
            return

        # 先获取所有 chunk 的 content
        content_hashes = [chunk.content_hash for chunk in blob.chunks]
        result = await self.session.execute(
            select(ChunkModel.content_hash, ChunkModel.content).where(
                ChunkModel.content_hash.in_(content_hashes)
            )
        )
        chunk_contents = {row.content_hash: row.content for row in result}
        logger.debug(f"Blob {blob.blob_name}: loaded {len(chunk_contents)} chunk contents")

        # 提取所有标识符
        symbol_values = []
        for chunk_ref in blob.chunks:
            content = chunk_contents.get(chunk_ref.content_hash)
            if not content:
                logger.warning(f"Blob {blob.blob_name}: chunk {chunk_ref.content_hash} content not found")
                continue

            symbols = self._symbol_extractor.extract_symbols(
                content=content,
                start_line=chunk_ref.start_line,
                end_line=chunk_ref.end_line,
            )

            for symbol in symbols:
                symbol_values.append(
                    {
                        "identifier": symbol.identifier,
                        "blob_name": blob.blob_name,
                        "content_hash": chunk_ref.content_hash,
                        "kind": symbol.kind,
                        "start_line": symbol.start_line,
                        "end_line": symbol.end_line,
                    }
                )

        logger.info(f"Blob {blob.blob_name}: extracted {len(symbol_values)} symbols from {len(blob.chunks)} chunks")

        if not symbol_values:
            return

        # 批量插入（冲突时忽略）
        stmt = self._insert()(SymbolOccurrenceModel).values(symbol_values)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["identifier", "blob_name", "content_hash", "kind"]
        )
        await self.session.execute(stmt)
        logger.info(f"Blob {blob.blob_name}: saved {len(symbol_values)} symbol occurrences")

    async def list_pending_names(self) -> list[str]:
        """全部 pending blob 名。队列对账要全集，且只需要标识不需要聚合。"""
        result = await self.session.execute(
            select(BlobModel.blob_name).where(BlobModel.status == "pending")
        )
        return list(result.scalars())

    async def find_stale_with_staging(
        self,
        stale_hours: int = 24,
        limit: int = 100,
    ) -> list[str]:
        """查找有 staging 但长时间未处理的 pending blob(用于重新入队或清理)"""
        from oce.infrastructure.persistence.models import BlobStagingModel
        from datetime import datetime, timedelta, timezone

        cutoff = datetime.now(timezone.utc) - timedelta(hours=stale_hours)
        result = await self.session.execute(
            select(BlobModel.blob_name)
            .join(BlobStagingModel, BlobStagingModel.blob_name == BlobModel.blob_name)
            .where(
                BlobModel.status == "pending",
                BlobStagingModel.created_at < cutoff,
            )
            .limit(limit)
        )
        return list(result.scalars())

    @staticmethod
    def _row_to_domain(row: BlobModel, chunks: list[ChunkRef]) -> Blob:
        return Blob(
            blob_name=row.blob_name,
            path=row.path,
            status=BlobStatus(row.status),
            chunks=chunks,
            content_size=row.content_size,
            language=row.language,
            file_type=row.file_type,
            retry_count=row.retry_count,
            last_seen=row.last_seen,
            created_at=row.created_at,
            error_message=row.error_message,
        )
