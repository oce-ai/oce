"""SQLAlchemy 工作单元。"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from oce.infrastructure.persistence.sql_blob_repo import SqlBlobRepository
from oce.infrastructure.persistence.sql_chain_repo import SqlChainRepository
from oce.infrastructure.persistence.sql_chunk_repo import SqlChunkRepository


class SqlAlchemyUnitOfWork:
    """为一个 application 用例提供同一事务内的仓储。"""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory
        self.session: AsyncSession | None = None

    async def __aenter__(self) -> "SqlAlchemyUnitOfWork":
        self.session = self._session_factory()
        self.blobs = SqlBlobRepository(self.session)
        self.chunks = SqlChunkRepository(self.session)
        self.chains = SqlChainRepository(self.session)
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        if self.session is None:
            return
        if exc_type is not None:
            await self.session.rollback()
        await self.session.close()

    async def commit(self) -> None:
        if self.session is None:
            raise RuntimeError("UnitOfWork is not active")
        await self.session.commit()

    async def rollback(self) -> None:
        if self.session is not None:
            await self.session.rollback()
