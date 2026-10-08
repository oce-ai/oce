"""Resolve and reload the active intent credential without interrupting predictions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlparse

from loguru import logger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from oce.domain.services.intent.port import IntentPrediction, IntentProvider, IntentProviderError
from oce.infrastructure.intent.openai_provider import OpenAIIntentProvider
from oce.infrastructure.intent.typesafe_provider import TypeSafeIntentProvider
from oce.infrastructure.llm.openai_compatible_client import OpenAICompatibleLLMClient, UsageCallback
from oce.infrastructure.persistence.models import ModelCredentialModel
from oce.shared.config.settings import LLMSettings, RetrievalSettings


@dataclass(frozen=True)
class IntentRuntimeConfig:
    transport: str
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float
    credential_id: int = 0
    proxy: str | None = None
    tpm_limit: int = 0


def _secret_value(value) -> str:
    return value.get_secret_value() if hasattr(value, "get_secret_value") else str(value or "")


def _is_typesafe_endpoint(endpoint: str) -> bool:
    host = (urlparse(endpoint).hostname or "").lower()
    return host == "typesafe.ai" or host.endswith(".typesafe.ai")


def _typesafe_base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    for suffix in ("/v1/systemone", "/v1"):
        if endpoint.endswith(suffix):
            return endpoint[:-len(suffix)]
    return endpoint


class CredentialConfiguredIntentProvider:
    def __init__(
        self,
        session_factory: Callable[[], AsyncSession],
        retrieval: RetrievalSettings,
        llm_fallback: LLMSettings,
        *,
        on_usage: UsageCallback | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._retrieval = retrieval
        self._llm_fallback = llm_fallback
        self._on_usage = on_usage
        self._delegate: IntentProvider | None = None
        self._config: IntentRuntimeConfig | None = None
        self._resolved = False
        self._lock = asyncio.Lock()
        self._reload_lock = asyncio.Lock()
        self._closed = False
        self._active_predictions = 0
        self._drained = asyncio.Event()
        self._drained.set()
        self._active_calls: dict[IntentProvider, int] = {}
        self._retired: set[IntentProvider] = set()
        self._usage_tasks: set[asyncio.Task] = set()

    async def predict(self, query: str) -> IntentPrediction:
        if self._closed:
            raise IntentProviderError("closed")
        self._active_predictions += 1
        self._drained.clear()
        delegate = None
        try:
            delegate = await self._acquire_delegate()
            return await delegate.predict(query)
        finally:
            try:
                if delegate is not None:
                    await self._release_delegate(delegate)
            finally:
                self._active_predictions -= 1
                if not self._active_predictions:
                    self._drained.set()

    async def _acquire_delegate(self) -> IntentProvider:
        async with self._lock:
            if self._delegate is None:
                if not self._resolved:
                    self._config = await self._resolve_config()
                    self._resolved = True
                if self._config is None:
                    raise IntentProviderError("no_credentials")
                self._delegate = self._build_delegate(self._config)
            delegate = self._delegate
            self._active_calls[delegate] = self._active_calls.get(delegate, 0) + 1
            return delegate

    async def _release_delegate(self, delegate: IntentProvider) -> None:
        close_previous = False
        async with self._lock:
            remaining = self._active_calls[delegate] - 1
            if remaining:
                self._active_calls[delegate] = remaining
            else:
                del self._active_calls[delegate]
                if delegate in self._retired:
                    self._retired.remove(delegate)
                    close_previous = True
        if close_previous:
            await self._close_delegate(delegate)

    async def _resolve_config(self) -> IntentRuntimeConfig | None:
        async with self._session_factory() as session:
            credential = (
                (await session.execute(
                    select(ModelCredentialModel)
                    .where(ModelCredentialModel.kind == "intent", ModelCredentialModel.status == "active")
                    .order_by(ModelCredentialModel.priority, ModelCredentialModel.id)
                    .limit(1)
                )).scalars().first()
            )
        retrieval = self._retrieval
        llm = self._llm_fallback
        if credential is not None and credential.api_key:
            endpoint = credential.endpoint or ""
            # An explicit endpoint is authoritative: legacy OpenAI keys must never
            # be sent to TypeSafe merely because the new default points there.
            typesafe = _is_typesafe_endpoint(endpoint) if endpoint else (
                (credential.provider or "").lower() == "typesafe"
                or (credential.model or "").startswith("jev-")
            )
            return IntentRuntimeConfig(
                transport="typesafe" if typesafe else "openai",
                api_key=credential.api_key,
                base_url=(endpoint or retrieval.intent_provider_base_url) if typesafe else (endpoint or llm.base_url),
                model=credential.model or (retrieval.intent_provider_model if typesafe else llm.model),
                timeout_seconds=float(credential.timeout_seconds),
                credential_id=credential.id,
                proxy=llm.proxy,
                tpm_limit=credential.tpm_limit if credential.tpm_limit is not None else llm.tpm_limit,
            )
        dedicated_key = _secret_value(retrieval.intent_provider_api_key).strip()
        if dedicated_key:
            return IntentRuntimeConfig(
                transport="typesafe",
                api_key=dedicated_key,
                base_url=retrieval.intent_provider_base_url,
                model=retrieval.intent_provider_model,
                timeout_seconds=retrieval.intent_provider_timeout_seconds,
            )
        legacy_key = _secret_value(llm.api_key).strip()
        if legacy_key:
            return IntentRuntimeConfig(
                transport="openai",
                api_key=legacy_key,
                base_url=llm.base_url,
                model=llm.model,
                timeout_seconds=retrieval.intent_provider_timeout_seconds,
                proxy=llm.proxy,
                tpm_limit=llm.tpm_limit,
            )
        return None

    def _build_delegate(self, config: IntentRuntimeConfig) -> IntentProvider:
        if config.transport == "typesafe":
            return TypeSafeIntentProvider(
                api_key=config.api_key,
                base_url=_typesafe_base_url(config.base_url),
                model=config.model,
                timeout_seconds=config.timeout_seconds,
                cache_size=self._retrieval.intent_provider_cache_size,
                on_usage=self._on_usage,
                credential_id=config.credential_id,
            )
        client = OpenAICompatibleLLMClient(
            api_key=config.api_key,
            base_url=config.base_url.rstrip("/"),
            timeout=config.timeout_seconds,
            proxy=config.proxy,
            tpm_limit=config.tpm_limit,
            on_usage=self._record_usage if self._on_usage is not None else None,
            credential_id=config.credential_id,
        )
        return OpenAIIntentProvider(client, model=config.model, timeout_seconds=config.timeout_seconds)

    async def _record_usage(self, credential_id: int, kind: str, model: str, prompt: int, completion: int) -> None:
        task = asyncio.create_task(self._report_usage(credential_id, kind, model, prompt, completion))
        self._usage_tasks.add(task)
        task.add_done_callback(self._usage_tasks.discard)

    async def _report_usage(self, credential_id: int, kind: str, model: str, prompt: int, completion: int) -> None:
        try:
            await self._on_usage(credential_id, kind, model, prompt, completion)
        except Exception as exc:
            logger.warning("intent usage collection failed: {}", type(exc).__name__)

    async def reload(self) -> int:
        async with self._reload_lock:
            if self._closed:
                raise IntentProviderError("closed")
            config = await self._resolve_config()
            replacement = self._build_delegate(config) if config is not None else None
            if self._closed:
                if replacement is not None:
                    await self._close_delegate(replacement)
                raise IntentProviderError("closed")
            close_previous = None
            async with self._lock:
                previous = self._delegate
                self._config = config
                self._resolved = True
                self._delegate = replacement
                if previous is not None:
                    if self._active_calls.get(previous, 0):
                        self._retired.add(previous)
                    else:
                        close_previous = previous
            if close_previous is not None:
                await self._close_delegate(close_previous)
            return int(replacement is not None)

    @staticmethod
    async def _close_delegate(delegate: IntentProvider) -> None:
        close = getattr(delegate, "aclose", None)
        if close is not None:
            await close()

    async def close(self) -> None:
        self._closed = True
        async with self._reload_lock:
            await self._drained.wait()
            async with self._lock:
                delegates = set(self._retired)
                if self._delegate is not None:
                    delegates.add(self._delegate)
                self._delegate = None
                self._retired.clear()
            await asyncio.gather(*(self._close_delegate(delegate) for delegate in delegates))

    async def aclose(self) -> None:
        await self.close()
