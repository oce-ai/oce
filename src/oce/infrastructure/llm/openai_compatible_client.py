"""OpenAI-compatible LLM client implementation."""
from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import httpx
from loguru import logger

from oce.infrastructure.llm.rate_limiter import TokenRateLimiter, estimate_tokens

# 用量回调：(credential_id, kind, model, prompt_tokens, completion_tokens)。
# credential_id 由上层（CredentialConfiguredLLMClient）按解析到的 DB 凭证注入；
# 无 DB 凭证（纯 env 回落）时为 0，sink 侧归一为 None。
UsageCallback = Callable[[int, str, str, int, int], Awaitable[None]]

# 输出 token 也计入 TPM。rerank 只回编号（实测约 34 token），按此量级留余量，
# 不按 max_tokens 记账，否则预算瞬间被 8000 占满。
_OUTPUT_TOKEN_ALLOWANCE = 256
# 限流器按估算值排队，估算偏松时仍可能 429，退避重试兜底
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 20.0

class OpenAICompatibleLLMClient:
    """OpenAI 兼容的 LLM 聊天客户端（/v1/chat/completions），rerank / rewrite / intent 共用。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: float = 120.0,
        proxy: str | None = None,
        tpm_limit: int | None = None,
        on_usage: UsageCallback | None = None,
        credential_id: int = 0,
    ):
        """
        Args:
            tpm_limit: 接口 TPM 上限。给定时请求前排队，避免 429 让上层静默降级。
            on_usage: 可选用量回调；每次成功 chat 后按真实 usage 上报，None 时零开销。
            credential_id: 解析到的 DB 凭证 id，随用量上报；纯 env 回落时为 0。
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.proxy = proxy
        self._on_usage = on_usage
        self._credential_id = credential_id
        self._limiter = (
            TokenRateLimiter(tpm_limit) if tpm_limit and tpm_limit > 0 else None
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "deepseek-v4-flash",
        temperature: float = 0.1,
        max_tokens: int = 8000,
        **kwargs,
    ) -> str:
        """
        发送聊天请求。

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            model: 模型名称
            temperature: 温度参数（0.1 = 更确定性）
            max_tokens: 最大生成 token 数。思维链也计入该预算，作为兜底放宽，
                避免推理输出把 content 挤空。

        Returns:
            模型响应内容
        """
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            **kwargs,
        }

        # 统一关闭思维链；调用方显式传 thinking 时（kwargs 已合并进 payload）不覆盖。
        if 'openrouter.ai' in self.base_url:
            payload.setdefault("reasoning", {
                "enabled": False,
                "effort": 'none',
            })
        else:
            payload.setdefault("thinking", {"type": "disabled"})

        client_kwargs: dict = {"timeout": self.timeout}
        if self.proxy:
            client_kwargs["proxy"] = self.proxy
            client_kwargs["verify"] = False

        # 输入按 prompt 估算，输出按固定余量记账
        estimated = (
            sum(estimate_tokens(m.get("content", "")) for m in messages)
            + _OUTPUT_TOKEN_ALLOWANCE
        )

        async with httpx.AsyncClient(**client_kwargs) as client:
            for attempt in range(1, _MAX_ATTEMPTS + 1):
                if self._limiter is not None:
                    waited = await self._limiter.acquire(estimated)
                    if waited > 0:
                        logger.debug(
                            "TPM limiter delayed request by {:.1f}s (est {} tokens)",
                            waited,
                            estimated,
                        )
                try:
                    response = await client.post(url, json=payload, headers=headers)
                    response.raise_for_status()
                    data = response.json()

                    # 提取响应内容
                    content = data["choices"][0]["message"]["content"]
                    if content is None:
                        content = data["choices"][0]["message"]["reasoning"]
                    # 按真实 usage 上报（旁路，缺字段则跳过）；credential_id 由构造时注入
                    if self._on_usage is not None:
                        usage = data.get("usage") or {}
                        prompt = int(usage.get("prompt_tokens", 0) or 0)
                        completion = int(usage.get("completion_tokens", 0) or 0)
                        if prompt or completion:
                            await self._on_usage(
                                self._credential_id, "llm", model, prompt, completion
                            )
                    return content

                except httpx.HTTPStatusError as e:
                    # 429 说明估算偏松或有其他调用方共用配额，退避后重试；
                    # 直接抛出会让 reranker 静默退回原始顺序。
                    if e.response.status_code == 429 and attempt < _MAX_ATTEMPTS:
                        logger.warning(
                            "LLM 429, retry {}/{} after {}s",
                            attempt,
                            _MAX_ATTEMPTS,
                            _RETRY_BACKOFF_SECONDS,
                        )
                        await asyncio.sleep(_RETRY_BACKOFF_SECONDS)
                        continue
                    logger.error(
                        f"LLM API error: {e.response.status_code} {e.response.text}"
                    )
                    raise
                except httpx.TimeoutException as e:
                    logger.error(f"LLM API timeout after {self.timeout}s: {e}")
                    raise
                except Exception as e:
                    logger.error(f"LLM client error: {type(e).__name__}: {e}")
                    raise

        raise RuntimeError("LLM chat exhausted retries without a response")
