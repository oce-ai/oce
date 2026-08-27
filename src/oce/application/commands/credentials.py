"""Embedding credential runtime commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from oce.application.messages import Command
from oce.shared.errors import ServiceNotReadyError


class ReloadableEmbeddingRuntime(Protocol):
    async def reload(self) -> int: ...


@dataclass(frozen=True)
class ReloadEmbeddingCredentialsCommand(Command):
    """Reload the active embedding credential and rebuild its client."""


@dataclass(frozen=True)
class ReloadEmbeddingCredentialsResult:
    reloaded: bool
    pool_size: int = 0
    reason: str | None = None


class ReloadEmbeddingCredentialsCommandHandler:
    def __init__(self, runtime: ReloadableEmbeddingRuntime) -> None:
        self._runtime = runtime

    async def handle(
        self,
        _command: ReloadEmbeddingCredentialsCommand,
    ) -> ReloadEmbeddingCredentialsResult:
        try:
            pool_size = await self._runtime.reload()
        except ServiceNotReadyError as exc:
            return ReloadEmbeddingCredentialsResult(False, reason=str(exc))
        return ReloadEmbeddingCredentialsResult(True, pool_size=pool_size)
