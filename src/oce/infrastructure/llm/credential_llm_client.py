"""按 kind 从 model_credentials 解析的懒加载 chat-LLM 客户端。

llm_rerank / query_rewrite / intent 各持一个实例（kind 不同），从凭证表解析自己的
active 凭证；取不到回落 LLMSettings（env）。实现 LLMClient.chat 协议，交给
domain 层的 reranker / rewriter / intent classifier 复用。

注意：底层 OpenAICompatibleLLMClient 每次 chat 内部新建 httpx client、调用间不持有
连接，故 reload 只需原子替换 delegate，无需关闭旧实例。三个 kind 各自独立限流：若共用
同一把 key，TPM 预算不共享（可接受的取舍，换取按用途独立管理/轮换）。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from oce.infrastructure.llm.openai_compatible_client import (
    OpenAICompatibleLLMClient,
    UsageCallback,
)
from oce.infrastructure.persistence.models import ModelCredentialModel
from oce.shared.config.settings import LLMSettings

# OpenAICompatibleLLMClient 的默认超时；env 回落分支无 DB timeout_seconds 时沿用。
_DEFAULT_LLM_TIMEOUT = 120.0


@dataclass(frozen=True)
class LLMRuntimeConfig:
    api_key: str
    base_url: str
    model: str | None
    proxy: str | None
    tpm_limit: int | None
    timeout_seconds: float
    credential_id: int = 0


class CredentialConfiguredLLMClient:
    """解析某个 kind 的 active 凭证并复用其 chat client。"""

    def __init__(
        self,
        kind: str,
        session_factory: Callable[[], AsyncSession],
        fallback: LLMSettings,
        *,
        fallback_model: str,
        on_usage: UsageCallback | None = None,
    ) -> None:
        self._kind = kind
        self._session_factory = session_factory
        self._fallback = fallback
        self._fallback_model = fallback_model
        self._on_usage = on_usage
        self._delegate: OpenAICompatibleLLMClient | None = None
        self._config: LLMRuntimeConfig | None = None
        self._lock = asyncio.Lock()

    async def chat(self, messages, model: str | None = None, **kwargs) -> str:
        delegate, config = await self._acquire()
        # 凭证 model 优先；其次调用方传入的 model；最后回落 env 默认模型。
        resolved = config.model or model or self._fallback_model
        return await delegate.chat(messages, model=resolved, **kwargs)

    async def _acquire(self) -> tuple[OpenAICompatibleLLMClient, LLMRuntimeConfig]:
        async with self._lock:
            if self._delegate is None:
                self._config = await self._resolve_config()
                self._delegate = self._build_delegate(self._config)
            return self._delegate, self._config

    async def _resolve_config(self) -> LLMRuntimeConfig:
        async with self._session_factory() as session:
            credential = (
                (
                    await session.execute(
                        select(ModelCredentialModel)
                        .where(
                            ModelCredentialModel.kind == self._kind,
                            ModelCredentialModel.status == "active",
                        )
                        .order_by(
                            ModelCredentialModel.priority,
                            ModelCredentialModel.id,
                        )
                        .limit(1)
                    )
                )
                .scalars()
                .first()
            )

        fb = self._fallback
        if credential is not None and credential.api_key:
            return LLMRuntimeConfig(
                api_key=credential.api_key,
                base_url=credential.endpoint or fb.base_url,
                model=credential.model,
                proxy=fb.proxy,
                tpm_limit=(
                    credential.tpm_limit
                    if credential.tpm_limit is not None
                    else fb.tpm_limit
                ),
                timeout_seconds=float(credential.timeout_seconds),
                credential_id=credential.id,
            )

        return LLMRuntimeConfig(
            api_key=fb.api_key.get_secret_value() if fb.api_key else "",
            base_url=fb.base_url,
            model=None,
            proxy=fb.proxy,
            tpm_limit=fb.tpm_limit,
            timeout_seconds=_DEFAULT_LLM_TIMEOUT,
            credential_id=0,
        )

    def _build_delegate(self, config: LLMRuntimeConfig) -> OpenAICompatibleLLMClient:
        return OpenAICompatibleLLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
            proxy=config.proxy,
            tpm_limit=config.tpm_limit,
            on_usage=self._on_usage,
            credential_id=config.credential_id,
        )

    async def reload(self) -> int:
        config = await self._resolve_config()
        delegate = self._build_delegate(config)
        async with self._lock:
            self._config = config
            self._delegate = delegate
        return 1
