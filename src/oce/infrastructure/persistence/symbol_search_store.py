"""基于 PostgreSQL symbol_occurrences 表的精确标识符召回。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from oce.domain.services.search import SearchHit
from oce.infrastructure.persistence.models import (
    BlobChunkModel,
    BlobModel,
    ChunkModel,
    SymbolOccurrenceModel,
)


class SymbolSearchStore:
    """通过 symbol_occurrences 倒排索引进行精确标识符召回。"""

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        max_scope_blobs: int = 2_000,
        timeout_seconds: float = 2.0,
    ) -> None:
        self._session_factory = session_factory
        self._max_scope_blobs = max_scope_blobs
        self._timeout_seconds = timeout_seconds

    async def search_exact(
        self,
        *,
        identifiers: Sequence[str],
        allowed_blob_names: Sequence[str] | None = None,
        top_k: int = 50,
    ) -> list[SearchHit]:
        """查询标识符出现位置。

        Args:
            identifiers: 标识符列表（函数名、类名等）
            allowed_blob_names: 允许的 blob 名称列表（scope 过滤）
            top_k: 最大返回数量

        Returns:
            SearchHit 列表，按 kind 优先级排序（endpoint > definition）
        """
        identifiers = tuple(dict.fromkeys(item for item in identifiers if item))
        if not identifiers or top_k <= 0:
            return []

        # 前置检查：scope 必须合理
        if allowed_blob_names is not None and self._max_scope_blobs > 0:
            if len(allowed_blob_names) > self._max_scope_blobs:
                return []

        try:
            async with asyncio.timeout(self._timeout_seconds):
                async with self._session_factory() as session:
                    return await self._query_symbols(
                        session=session,
                        identifiers=identifiers,
                        allowed_blob_names=frozenset(allowed_blob_names) if allowed_blob_names else None,
                        top_k=top_k,
                    )
        except TimeoutError:
            return []

    async def _query_symbols(
        self,
        session: AsyncSession,
        identifiers: Sequence[str],
        allowed_blob_names: frozenset[str] | None,
        top_k: int,
    ) -> list[SearchHit]:
        """执行实际查询。"""
        # 构建查询：从 symbol_occurrences 开始
        stmt = (
            select(
                SymbolOccurrenceModel.content_hash,
                SymbolOccurrenceModel.identifier,
                SymbolOccurrenceModel.kind,
                SymbolOccurrenceModel.blob_name,
                BlobModel.path,
                ChunkModel.content,
                BlobChunkModel.start_line,
                BlobChunkModel.end_line,
            )
            .join(ChunkModel, SymbolOccurrenceModel.content_hash == ChunkModel.content_hash)
            .join(BlobModel, SymbolOccurrenceModel.blob_name == BlobModel.blob_name)
            .join(
                BlobChunkModel,
                (BlobChunkModel.content_hash == SymbolOccurrenceModel.content_hash)
                & (BlobChunkModel.blob_name == SymbolOccurrenceModel.blob_name),
            )
            .where(SymbolOccurrenceModel.identifier.in_(identifiers))
        )

        # Scope 过滤
        if allowed_blob_names:
            stmt = stmt.where(SymbolOccurrenceModel.blob_name.in_(allowed_blob_names))

        stmt = stmt.limit(max(top_k * 20, top_k))

        result = await session.execute(stmt)
        rows = result.all()

        # 构建 SearchHit 并排序
        hits = []
        seen_hashes = set()

        for content_hash, identifier, kind, blob_name, path, content, start_line, end_line in rows:
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)

            # 按 kind 分配分数
            score = self._score_by_kind(kind)

            hits.append(
                SearchHit(
                    blob_name=blob_name,
                    path=path,
                    content=content,
                    score=score,
                    content_hash=content_hash,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

        # 按分数降序排序，取 top_k
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    @staticmethod
    def _score_by_kind(kind: str) -> float:
        """按 kind 分配优先级分数。"""
        if kind == "endpoint":
            return 1.0
        elif kind == "definition":
            return 0.95
        else:
            return 0.85
