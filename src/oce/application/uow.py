"""Application 层使用的工作单元协议。"""

from __future__ import annotations

from typing import Protocol

from oce.domain.repositories import BlobRepository, ChainRepository, ChunkRepository


class UnitOfWork(Protocol):
    blobs: BlobRepository
    chunks: ChunkRepository
    chains: ChainRepository

    async def __aenter__(self) -> "UnitOfWork": ...

    async def __aexit__(self, exc_type, exc, traceback) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


class UnitOfWorkFactory(Protocol):
    def __call__(self) -> UnitOfWork: ...
