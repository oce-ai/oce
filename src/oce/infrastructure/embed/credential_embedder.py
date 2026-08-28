"""Lazy embedding runtime configured by the active database credential."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from oce.infrastructure.embed.openai_embedder import OpenAIEmbedder, UsageCallback
from oce.infrastructure.persistence.models import EmbeddingCredentialModel
from oce.shared.config.settings import EmbeddingSettings
from oce.shared.errors import ServiceNotReadyError


@dataclass(frozen=True)
class EmbeddingRuntimeConfig:
    endpoint: str
    api_key: str
    model: str
    dimensions: int
    max_batch_size: int
    max_batch_chars: int
    max_input_chars: int
    input_overlap_chars: int
    max_concurrency: int
    timeout_seconds: float
    proxy: str | None
    credential_id: int = 0


class CredentialConfiguredEmbedder:
    """Resolve one active credential, then reuse its OpenAI-compatible client."""

    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        fallback: EmbeddingSettings,
        *,
        expected_dimensions: int,
        on_usage: UsageCallback | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._fallback = fallback
        self._expected_dimensions = expected_dimensions
        self._on_usage = on_usage
        self._delegate: OpenAIEmbedder | None = None
        self._lock = asyncio.Lock()
        self._active_calls: dict[OpenAIEmbedder, int] = {}
        self._retired: set[OpenAIEmbedder] = set()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        delegate = await self._acquire_delegate()
        try:
            return await delegate.embed_documents(texts)
        finally:
            await self._release_delegate(delegate)

    async def embed_query(self, text: str) -> list[float]:
        delegate = await self._acquire_delegate()
        try:
            return await delegate.embed_query(text)
        finally:
            await self._release_delegate(delegate)

    async def _acquire_delegate(self) -> OpenAIEmbedder:
        async with self._lock:
            if self._delegate is None:
                config = await self._resolve_config()
                self._delegate = self._build_delegate(config)
            delegate = self._delegate
            self._active_calls[delegate] = self._active_calls.get(delegate, 0) + 1
            return delegate

    async def _release_delegate(self, delegate: OpenAIEmbedder) -> None:
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
            await delegate.close()

    def _build_delegate(self, config: EmbeddingRuntimeConfig) -> OpenAIEmbedder:
        return OpenAIEmbedder.from_endpoint(
            endpoint=config.endpoint,
            api_key=config.api_key,
            model=config.model,
            dimensions=config.dimensions,
            max_batch_size=config.max_batch_size,
            max_batch_chars=config.max_batch_chars,
            max_input_chars=config.max_input_chars,
            input_overlap_chars=config.input_overlap_chars,
            max_concurrency=config.max_concurrency,
            timeout=config.timeout_seconds,
            proxy=config.proxy,
            credential_id=config.credential_id,
            on_usage=self._on_usage,
        )

    async def _resolve_config(self) -> EmbeddingRuntimeConfig:
        async with self._session_factory() as session:
            credential = (
                (
                    await session.execute(
                        select(EmbeddingCredentialModel)
                        .where(
                            EmbeddingCredentialModel.status == "active",
                            EmbeddingCredentialModel.embed_endpoint.is_not(None),
                            EmbeddingCredentialModel.embed_model.is_not(None),
                        )
                        .order_by(
                            EmbeddingCredentialModel.priority,
                            EmbeddingCredentialModel.id,
                        )
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )

        if credential is None:
            key = (
                self._fallback.api_key.get_secret_value()
                if self._fallback.api_key is not None
                else ""
            )
            if not key:
                raise ServiceNotReadyError(
                    "No active embedding credential or EMBED_API_KEY is configured"
                )
            config = EmbeddingRuntimeConfig(
                endpoint=self._fallback.endpoint,
                api_key=key,
                model=self._fallback.model,
                dimensions=self._fallback.dimensions,
                max_batch_size=self._fallback.max_batch_size,
                max_batch_chars=self._fallback.max_batch_chars,
                max_input_chars=self._fallback.max_input_chars,
                input_overlap_chars=self._fallback.input_overlap_chars,
                max_concurrency=self._fallback.max_concurrency,
                timeout_seconds=self._fallback.timeout_seconds,
                proxy=self._fallback.proxy,
            )
        else:
            config = EmbeddingRuntimeConfig(
                endpoint=credential.embed_endpoint,
                api_key=credential.api_key,
                model=credential.embed_model,
                dimensions=credential.dimensions,
                max_batch_size=credential.max_batch_size,
                max_batch_chars=credential.max_batch_chars,
                max_input_chars=credential.max_input_chars,
                input_overlap_chars=credential.input_overlap_chars,
                max_concurrency=self._fallback.max_concurrency,
                timeout_seconds=float(credential.timeout_seconds),
                proxy=self._fallback.proxy,
                credential_id=credential.id,
            )

        if config.dimensions != self._expected_dimensions:
            raise ServiceNotReadyError(
                "Embedding credential dimensions do not match MILVUS_DENSE_DIM"
            )
        return config

    async def close(self) -> None:
        async with self._lock:
            delegates = set(self._retired)
            if self._delegate is not None:
                delegates.add(self._delegate)
            self._delegate = None
            self._retired.clear()
        await asyncio.gather(*(delegate.close() for delegate in delegates))

    async def reload(self) -> int:
        replacement = await self.prepare_reload()
        return await self.activate_prepared(replacement)

    async def prepare_reload(self) -> OpenAIEmbedder:
        return self._build_delegate(await self._resolve_config())

    async def activate_prepared(self, replacement: OpenAIEmbedder) -> int:
        close_previous: OpenAIEmbedder | None = None
        async with self._lock:
            previous = self._delegate
            self._delegate = replacement
            if previous is not None:
                if self._active_calls.get(previous, 0):
                    self._retired.add(previous)
                else:
                    close_previous = previous
        if close_previous is not None:
            await close_previous.close()
        return 1

    async def discard_prepared(self, replacement: OpenAIEmbedder) -> None:
        await replacement.close()
