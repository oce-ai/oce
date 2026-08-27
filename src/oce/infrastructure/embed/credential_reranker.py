"""Lazy rerank runtime configured by the active database credential."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from oce.domain.services.reranker import NoopReranker, Reranker
from oce.infrastructure.embed.openai_reranker import OpenAIReranker
from oce.infrastructure.persistence.models import EmbeddingCredentialModel, EmbeddingProviderModel
from oce.shared.config.settings import RerankSettings


@dataclass(frozen=True)
class RerankRuntimeConfig:
    endpoint: str
    api_key: str
    model: str
    top_n: int
    min_score: float
    timeout_seconds: float


class CredentialConfiguredReranker:
    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        fallback: RerankSettings,
        *,
        fallback_embedding_key: str | None,
    ) -> None:
        self._session_factory = session_factory
        self._fallback = fallback
        self._fallback_embedding_key = fallback_embedding_key
        self._delegate: Reranker | None = None
        self._lock = asyncio.Lock()
        self._active_calls: dict[Reranker, int] = {}
        self._retired: set[Reranker] = set()

    async def rerank(self, query: str, hits):
        delegate = await self._acquire_delegate()
        try:
            return await delegate.rerank(query, hits)
        finally:
            await self._release_delegate(delegate)

    async def _acquire_delegate(self) -> Reranker:
        async with self._lock:
            if self._delegate is None:
                self._delegate = self._build_delegate(await self._resolve_config())
            delegate = self._delegate
            self._active_calls[delegate] = self._active_calls.get(delegate, 0) + 1
            return delegate

    async def _release_delegate(self, delegate: Reranker) -> None:
        close_delegate = False
        async with self._lock:
            remaining = self._active_calls[delegate] - 1
            if remaining:
                self._active_calls[delegate] = remaining
            else:
                del self._active_calls[delegate]
                if delegate in self._retired:
                    self._retired.remove(delegate)
                    close_delegate = True
        if close_delegate:
            await self._close_delegate(delegate)

    async def _resolve_config(self) -> RerankRuntimeConfig | None:
        if not self._fallback.enabled:
            return None
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(EmbeddingCredentialModel, EmbeddingProviderModel)
                    .join(
                        EmbeddingProviderModel,
                        EmbeddingProviderModel.id == EmbeddingCredentialModel.provider_id,
                    )
                    .where(
                        EmbeddingCredentialModel.status == "active",
                        EmbeddingProviderModel.rerank_endpoint.is_not(None),
                        EmbeddingProviderModel.rerank_model.is_not(None),
                    )
                    .order_by(
                        EmbeddingCredentialModel.priority,
                        EmbeddingCredentialModel.id,
                    )
                    .limit(1)
                )
            ).first()

        if row is not None:
            credential, provider = row
            return RerankRuntimeConfig(
                endpoint=provider.rerank_endpoint,
                api_key=credential.api_key,
                model=provider.rerank_model,
                top_n=self._fallback.top_n,
                min_score=self._fallback.min_score,
                timeout_seconds=float(credential.timeout_seconds),
            )

        key = (
            self._fallback.api_key.get_secret_value()
            if self._fallback.api_key is not None
            else self._fallback_embedding_key
        )
        if not key:
            return None
        return RerankRuntimeConfig(
            endpoint=self._fallback.endpoint,
            api_key=key,
            model=self._fallback.model,
            top_n=self._fallback.top_n,
            min_score=self._fallback.min_score,
            timeout_seconds=self._fallback.timeout_seconds,
        )

    @staticmethod
    def _build_delegate(config: RerankRuntimeConfig | None) -> Reranker:
        if config is None:
            return NoopReranker()
        return OpenAIReranker(
            endpoint=config.endpoint,
            api_key=config.api_key,
            model=config.model,
            top_n=config.top_n,
            min_score=config.min_score,
            timeout=config.timeout_seconds,
        )

    @staticmethod
    async def _close_delegate(delegate: Reranker) -> None:
        close = getattr(delegate, "close", None)
        if close is not None:
            await close()

    async def reload(self) -> None:
        replacement = await self.prepare_reload()
        await self.activate_prepared(replacement)

    async def prepare_reload(self) -> Reranker:
        return self._build_delegate(await self._resolve_config())

    async def activate_prepared(self, replacement: Reranker) -> None:
        close_previous: Reranker | None = None
        async with self._lock:
            previous = self._delegate
            self._delegate = replacement
            if previous is not None:
                if self._active_calls.get(previous, 0):
                    self._retired.add(previous)
                else:
                    close_previous = previous
        if close_previous is not None:
            await self._close_delegate(close_previous)

    async def discard_prepared(self, replacement: Reranker) -> None:
        await self._close_delegate(replacement)

    async def close(self) -> None:
        async with self._lock:
            delegates = set(self._retired)
            if self._delegate is not None:
                delegates.add(self._delegate)
            self._delegate = None
            self._retired.clear()
        await asyncio.gather(*(self._close_delegate(item) for item in delegates))
