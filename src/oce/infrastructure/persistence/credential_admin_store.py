"""凭据 admin CRUD 的 SQL 实现（CredentialAdminStore 端口）。"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from oce.application.credential_admin import (
    CredentialCreate,
    CredentialRecord,
    CredentialUpdate,
)
from oce.infrastructure.persistence.models import EmbeddingCredentialModel
from oce.shared.errors import CredentialConflictError

# CredentialUpdate 中可直接透传到模型的标量字段（api_key 单独处理以同步 hash）。
_UPDATABLE_FIELDS = (
    "name",
    "provider",
    "status",
    "priority",
    "embed_endpoint",
    "embed_model",
    "dimensions",
    "max_batch_size",
    "max_batch_chars",
    "max_input_chars",
    "input_overlap_chars",
    "rerank_endpoint",
    "rerank_model",
    "timeout_seconds",
    "rate_limit",
    "note",
)


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _to_record(model: EmbeddingCredentialModel) -> CredentialRecord:
    return CredentialRecord(
        id=model.id,
        provider=model.provider,
        name=model.name,
        status=model.status,
        priority=model.priority,
        embed_endpoint=model.embed_endpoint,
        embed_model=model.embed_model,
        dimensions=model.dimensions,
        max_batch_size=model.max_batch_size,
        max_batch_chars=model.max_batch_chars,
        max_input_chars=model.max_input_chars,
        input_overlap_chars=model.input_overlap_chars,
        rerank_endpoint=model.rerank_endpoint,
        rerank_model=model.rerank_model,
        timeout_seconds=model.timeout_seconds,
        rate_limit=model.rate_limit,
        note=model.note,
        api_key_last4=(model.api_key or "")[-4:],
        last_used_at=model.last_used_at,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


class SqlCredentialAdminStore:
    def __init__(self, session_factory: Callable[[], AsyncSession]) -> None:
        self._session_factory = session_factory

    async def list(self) -> list[CredentialRecord]:
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(EmbeddingCredentialModel).order_by(
                        EmbeddingCredentialModel.priority,
                        EmbeddingCredentialModel.id,
                    )
                )
            ).scalars().all()
            return [_to_record(row) for row in rows]

    async def create(self, data: CredentialCreate) -> CredentialRecord:
        model = EmbeddingCredentialModel(
            provider=data.provider,
            name=data.name,
            api_key=data.api_key,
            api_key_hash=_hash_key(data.api_key),
            status=data.status,
            priority=data.priority,
            embed_endpoint=data.embed_endpoint,
            embed_model=data.embed_model,
            dimensions=data.dimensions,
            max_batch_size=data.max_batch_size,
            max_batch_chars=data.max_batch_chars,
            max_input_chars=data.max_input_chars,
            input_overlap_chars=data.input_overlap_chars,
            rerank_endpoint=data.rerank_endpoint,
            rerank_model=data.rerank_model,
            timeout_seconds=data.timeout_seconds,
            rate_limit=data.rate_limit,
            note=data.note,
        )
        return await self._persist_new(model)

    async def update(
        self, credential_id: int, changes: CredentialUpdate
    ) -> CredentialRecord | None:
        async with self._session_factory() as session:
            model = await session.get(EmbeddingCredentialModel, credential_id)
            if model is None:
                return None
            for field in _UPDATABLE_FIELDS:
                value = getattr(changes, field)
                if value is not None:
                    setattr(model, field, value)
            if changes.api_key is not None:
                model.api_key = changes.api_key
                model.api_key_hash = _hash_key(changes.api_key)
            model.updated_at = datetime.now(timezone.utc)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise CredentialConflictError() from exc
            await session.refresh(model)
            return _to_record(model)

    async def delete(self, credential_id: int) -> bool:
        async with self._session_factory() as session:
            model = await session.get(EmbeddingCredentialModel, credential_id)
            if model is None:
                return False
            await session.delete(model)
            await session.commit()
            return True

    async def duplicate(
        self, credential_id: int, *, name: str, api_key: str
    ) -> CredentialRecord | None:
        async with self._session_factory() as session:
            src = await session.get(EmbeddingCredentialModel, credential_id)
            if src is None:
                return None
            clone = EmbeddingCredentialModel(
                provider=src.provider,
                name=name,
                api_key=api_key,
                api_key_hash=_hash_key(api_key),
                status=src.status,
                priority=src.priority,
                embed_endpoint=src.embed_endpoint,
                embed_model=src.embed_model,
                dimensions=src.dimensions,
                max_batch_size=src.max_batch_size,
                max_batch_chars=src.max_batch_chars,
                max_input_chars=src.max_input_chars,
                input_overlap_chars=src.input_overlap_chars,
                rerank_endpoint=src.rerank_endpoint,
                rerank_model=src.rerank_model,
                timeout_seconds=src.timeout_seconds,
                rate_limit=src.rate_limit,
                note=src.note,
            )
        return await self._persist_new(clone)

    async def _persist_new(
        self, model: EmbeddingCredentialModel
    ) -> CredentialRecord:
        async with self._session_factory() as session:
            session.add(model)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                raise CredentialConflictError() from exc
            await session.refresh(model)
            return _to_record(model)
