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
    CredentialDuplicate,
    CredentialRecord,
    CredentialUpdate,
)
from oce.infrastructure.persistence.models import ModelCredentialModel
from oce.shared.errors import CredentialConflictError

# CredentialCreate/Update 中可直接透传到模型的标量字段（api_key 单独处理以同步 hash）。
_SCALAR_FIELDS = (
    "kind",
    "provider",
    "name",
    "status",
    "priority",
    "endpoint",
    "model",
    "timeout_seconds",
    "rate_limit",
    "note",
    "dimensions",
    "max_batch_size",
    "max_batch_chars",
    "max_input_chars",
    "input_overlap_chars",
    "top_n",
    "min_score",
    "tpm_limit",
    "max_candidates",
    "output_top_k",
    "snippet_chars",
    "num_rewrites",
)

# update 场景不允许把 kind 之外的主键类字段清空为默认；name/kind 必填不可置 None。
_UPDATABLE_FIELDS = _SCALAR_FIELDS


def _hash_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _to_record(model: ModelCredentialModel) -> CredentialRecord:
    return CredentialRecord(
        id=model.id,
        kind=model.kind,
        provider=model.provider,
        name=model.name,
        status=model.status,
        priority=model.priority,
        endpoint=model.endpoint,
        model=model.model,
        timeout_seconds=model.timeout_seconds,
        rate_limit=model.rate_limit,
        note=model.note,
        dimensions=model.dimensions,
        max_batch_size=model.max_batch_size,
        max_batch_chars=model.max_batch_chars,
        max_input_chars=model.max_input_chars,
        input_overlap_chars=model.input_overlap_chars,
        top_n=model.top_n,
        min_score=model.min_score,
        tpm_limit=model.tpm_limit,
        max_candidates=model.max_candidates,
        output_top_k=model.output_top_k,
        snippet_chars=model.snippet_chars,
        num_rewrites=model.num_rewrites,
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
                    select(ModelCredentialModel).order_by(
                        ModelCredentialModel.kind,
                        ModelCredentialModel.priority,
                        ModelCredentialModel.id,
                    )
                )
            ).scalars().all()
            return [_to_record(row) for row in rows]

    async def create(self, data: CredentialCreate) -> CredentialRecord:
        model = ModelCredentialModel(
            api_key=data.api_key,
            api_key_hash=_hash_key(data.api_key),
            **{field: getattr(data, field) for field in _SCALAR_FIELDS},
        )
        return await self._persist_new(model)

    async def update(
        self, credential_id: int, changes: CredentialUpdate
    ) -> CredentialRecord | None:
        async with self._session_factory() as session:
            model = await session.get(ModelCredentialModel, credential_id)
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
            model = await session.get(ModelCredentialModel, credential_id)
            if model is None:
                return False
            await session.delete(model)
            await session.commit()
            return True

    async def duplicate(
        self, credential_id: int, changes: CredentialDuplicate
    ) -> CredentialRecord | None:
        async with self._session_factory() as session:
            src = await session.get(ModelCredentialModel, credential_id)
            if src is None:
                return None
            # 先继承源行全部标量字段，再用非 None 的覆盖字段替换（name 也走覆盖）。
            values = {field: getattr(src, field) for field in _SCALAR_FIELDS}
            for field in _SCALAR_FIELDS:
                override = getattr(changes, field)
                if override is not None:
                    values[field] = override
            # api_key 省略即复用源 key，这正是“同一把 key 换用途”复制的关键。
            api_key = changes.api_key if changes.api_key is not None else src.api_key
            clone = ModelCredentialModel(
                api_key=api_key,
                api_key_hash=_hash_key(api_key),
                **values,
            )
        return await self._persist_new(clone)

    async def _persist_new(
        self, model: ModelCredentialModel
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
