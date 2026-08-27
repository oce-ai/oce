"""基于 PostgreSQL/SQLite 元数据的精确代码标识符召回。"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Sequence

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from oce.domain.services.search import SearchHit
from oce.infrastructure.persistence.models import (
    BlobChunkModel,
    BlobModel,
    ChunkModel,
)


class SqlExactSearchStore:
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
        identifiers = tuple(dict.fromkeys(item for item in identifiers if item))
        if not identifiers or top_k <= 0:
            return []
        if (
            allowed_blob_names is None
            or not allowed_blob_names
            or self._max_scope_blobs == 0
            or len(allowed_blob_names) > self._max_scope_blobs
        ):
            return []

        stmt = (
            select(
                BlobModel.blob_name,
                BlobModel.path,
                ChunkModel.content_hash,
                ChunkModel.content,
                BlobChunkModel.start_line,
                BlobChunkModel.end_line,
            )
            .join(BlobChunkModel, BlobChunkModel.blob_name == BlobModel.blob_name)
            .join(ChunkModel, ChunkModel.content_hash == BlobChunkModel.content_hash)
            .where(
                BlobModel.status == "ready",
                or_(
                    *(ChunkModel.content.contains(item, autoescape=True) for item in identifiers)
                ),
            )
        )
        stmt = stmt.where(BlobModel.blob_name.in_(allowed_blob_names))
        stmt = stmt.limit(max(top_k * 20, top_k))

        async with self._session_factory() as session:
            async with asyncio.timeout(self._timeout_seconds):
                rows = (await session.execute(stmt)).all()

        ranked = sorted(
            rows,
            key=lambda row: self._rank(row.content, identifiers),
            reverse=True,
        )
        return [
            SearchHit(
                blob_name=row.blob_name,
                path=row.path,
                content_hash=row.content_hash,
                content=row.content,
                start_line=row.start_line,
                end_line=row.end_line,
                score=self._rank(row.content, identifiers),
            )
            for row in ranked[:top_k]
        ]

    @staticmethod
    def _rank(content: str, identifiers: Sequence[str]) -> float:
        best = 0.75
        for identifier in identifiers:
            escaped = re.escape(identifier.rsplit("::", 1)[-1])
            endpoint_declaration = re.compile(
                rf"(?ms)(?:#\[(?:tauri::command|pytauri::command)[^]]*\]|"
                rf"@(?:app|router)\.(?:get|post|put|patch|delete)\([^\n]*\))"
                rf"\s*(?:(?:pub|export)(?:\([^)]*\))?\s+)?"
                rf"(?:(?:async|default)\s+)?(?:fn|def|function)\s+{escaped}\b"
            )
            declaration = re.compile(
                rf"(?m)^\s*(?:(?:pub|export)(?:\([^)]*\))?\s+)?"
                rf"(?:(?:async|default)\s+)?"
                rf"(?:fn|def|function|class|interface|type|struct|enum|trait|const|static)"
                rf"\s+{escaped}\b"
            )
            assignment = re.compile(
                rf"(?m)^\s*(?:export\s+)?(?:const|let|var)\s+{escaped}\s*="
            )
            if endpoint_declaration.search(content):
                return 1.0
            if declaration.search(content) or assignment.search(content):
                return 0.95
            if re.search(rf"(?<![A-Za-z0-9_$]){re.escape(identifier)}(?![A-Za-z0-9_$])", content):
                best = max(best, 0.85)
        return best