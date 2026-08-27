"""SQLAlchemy checkpoint chain 仓储。"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Sequence

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from oce.domain.chain.chain import Chain
from oce.infrastructure.persistence.models import BlobModel, ChainMemberModel, ChainModel
from oce.domain.repositories import ChainRepository


_MEMBER_WRITE_BATCH_SIZE = 1_000


class SqlChainRepository(ChainRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    def _insert(self):
        bind = self.session.get_bind()
        return sqlite_insert if bind.dialect.name == "sqlite" else pg_insert

    async def get(self, chain_id: str) -> Chain | None:
        row = (
            await self.session.execute(
                select(ChainModel).where(ChainModel.chain_id == chain_id)
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return Chain(
            chain_id=row.chain_id,
            version=row.version,
            members=await self.get_members(chain_id),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    async def exists(self, chain_id: str) -> bool:
        value = await self.session.scalar(
            select(func.count()).select_from(ChainModel).where(ChainModel.chain_id == chain_id)
        )
        return bool(value)

    async def create(self, members: Sequence[str]) -> Chain:
        chain_id = uuid.uuid4().hex
        unique_members = sorted(set(members))
        now = datetime.now(timezone.utc)
        await self.session.execute(
            self._insert()(ChainModel).values(
                chain_id=chain_id,
                version=1,
                total_blobs=len(unique_members),
                created_at=now,
                updated_at=now,
            )
        )
        if unique_members:
            await self._insert_members(chain_id, unique_members)
        return Chain(chain_id=chain_id, version=1, members=set(unique_members))

    async def get_members(self, chain_id: str) -> set[str]:
        rows = await self.session.execute(
            select(ChainMemberModel.blob_name).where(ChainMemberModel.chain_id == chain_id)
        )
        return set(rows.scalars())

    async def apply_checkpoint(
        self,
        chain_id: str,
        added: Sequence[str],
        deleted: Sequence[str],
    ) -> int | None:
        current_version = await self.session.scalar(
            select(ChainModel.version).where(ChainModel.chain_id == chain_id)
        )
        if current_version is None:
            return None

        unique_deleted = sorted(set(deleted))
        for offset in range(0, len(unique_deleted), _MEMBER_WRITE_BATCH_SIZE):
            await self.session.execute(
                delete(ChainMemberModel).where(
                    ChainMemberModel.chain_id == chain_id,
                    ChainMemberModel.blob_name.in_(
                        unique_deleted[offset : offset + _MEMBER_WRITE_BATCH_SIZE]
                    ),
                )
            )
        if added:
            await self._insert_members(chain_id, sorted(set(added)))

        count = await self.session.scalar(
            select(func.count())
            .select_from(ChainMemberModel)
            .where(ChainMemberModel.chain_id == chain_id)
        )
        new_version = int(current_version) + 1
        await self.session.execute(
            update(ChainModel)
            .where(ChainModel.chain_id == chain_id)
            .values(
                version=new_version,
                total_blobs=int(count or 0),
                updated_at=datetime.now(timezone.utc),
            )
        )
        return new_version

    async def _insert_members(self, chain_id: str, members: Sequence[str]) -> None:
        for offset in range(0, len(members), _MEMBER_WRITE_BATCH_SIZE):
            values = [
                {"chain_id": chain_id, "blob_name": name}
                for name in members[offset : offset + _MEMBER_WRITE_BATCH_SIZE]
            ]
            stmt = self._insert()(ChainMemberModel).values(values)
            await self.session.execute(
                stmt.on_conflict_do_nothing(index_elements=["chain_id", "blob_name"])
            )

    async def touch_members(self, chain_id: str) -> None:
        member_names = select(ChainMemberModel.blob_name).where(
            ChainMemberModel.chain_id == chain_id
        )
        await self.session.execute(
            update(BlobModel)
            .where(BlobModel.blob_name.in_(member_names))
            .values(last_seen=datetime.now(timezone.utc))
        )

    async def delete(self, chain_id: str) -> None:
        await self.session.execute(
            delete(ChainMemberModel).where(ChainMemberModel.chain_id == chain_id)
        )
        await self.session.execute(delete(ChainModel).where(ChainModel.chain_id == chain_id))

    async def find_expired(self, ttl_days: int) -> list[str]:
        threshold = datetime.now(timezone.utc) - timedelta(days=ttl_days)
        rows = await self.session.execute(
            select(ChainModel.chain_id).where(ChainModel.updated_at < threshold)
        )
        return list(rows.scalars())
