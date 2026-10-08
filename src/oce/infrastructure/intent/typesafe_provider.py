"""TypeSafe System One 意图判定适配器。

按官方契约调用：

    POST {base_url}/v1/systemone
    Authorization: Bearer <api_key>
    {"state": <query>, "model": <model>, "questions": {"intent": {
        "type": "choice", "instructions": ..., "criteria": {<label>: <desc>}}}}

响应里读取 ``answers.intent`` 的 ``choice`` / ``probabilities`` /
``confidence``。Choice 的 criteria 是每次请求下发的，所以增删标签不需要
重新训练模型，这也是本次把标签体系扩到 8 类的可行性基础。

设计约束：
- 一切失败都转成 `IntentProviderError`，绝不把异常泄漏给检索流程；
- 无 API key 时构造期即拒绝，调用方据此完全不注入 provider；
- 所有参数来自配置，无硬编码超时；
- 带上界的规范化查询缓存，避免重复付费。
"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from hashlib import sha256
from time import perf_counter
from typing import Any, Mapping

import httpx
from loguru import logger

from oce.domain.services.intent.port import (
    IntentPrediction,
    IntentProviderError,
)
from oce.domain.services.intent.taxonomy import (
    INSTRUCTIONS,
    choice_criteria,
    intent_from_label,
)
from oce.infrastructure.llm.openai_compatible_client import UsageCallback

#: 官方评估端点的路径后缀。
SYSTEM_ONE_PATH = "/v1/systemone"
#: 请求里 question 的键名；答案在同名键下返回。
QUESTION_ID = "intent"


class TypeSafeIntentProvider:
    """通过 TypeSafe System One 的 Choice 原语判定查询意图。"""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float,
        cache_size: int = 1024,
        client: httpx.AsyncClient | None = None,
        on_usage: UsageCallback | None = None,
        credential_id: int = 0,
    ) -> None:
        if not (api_key or "").strip():
            # 缺 key 时不应存在 provider 实例：调用方以纯规则模式运行。
            raise ValueError("TypeSafe API key is required to build the intent provider")
        self._api_key = api_key.strip()
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self.cache_size = max(0, int(cache_size))
        self._client = client
        self._owns_client = client is None
        self._cache: OrderedDict[str, IntentPrediction] = OrderedDict()
        self._on_usage = on_usage
        self._credential_id = credential_id
        self._closed = False
        self._active_predictions = 0
        self._drained = asyncio.Event()
        self._drained.set()
        self._close_lock = asyncio.Lock()
        self._usage_tasks: set[asyncio.Task] = set()

    # ── 生命周期 ────────────────────────────────────────────────────────
    async def aclose(self) -> None:
        self._closed = True
        async with self._close_lock:
            await self._drained.wait()
            if self._owns_client and self._client is not None:
                await self._client.aclose()
                self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_seconds)
        return self._client

    # ── 缓存 ────────────────────────────────────────────────────────────
    @property
    def cache_size_used(self) -> int:
        return len(self._cache)

    def _cache_key(self, query: str) -> str:
        normalized = " ".join((query or "").split()).casefold()
        material = f"{normalized}\n{self.model}\n{sorted(choice_criteria())}"
        return sha256(material.encode("utf-8")).hexdigest()

    def _cache_put(self, key: str, value: IntentPrediction) -> None:
        if not self.cache_size:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    # ── 请求 ────────────────────────────────────────────────────────────
    def _payload(self, query: str) -> dict[str, Any]:
        return {
            "state": query,
            "model": self.model,
            "questions": {
                QUESTION_ID: {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": choice_criteria(),
                }
            },
        }

    async def predict(self, query: str) -> IntentPrediction:
        if self._closed:
            raise IntentProviderError("closed")
        # Register before the first await so shutdown also drains callers waiting
        # to enter the transport, rather than closing their shared client early.
        self._active_predictions += 1
        self._drained.clear()
        try:
            return await self._predict(query)
        finally:
            self._active_predictions -= 1
            if not self._active_predictions:
                self._drained.set()

    async def _predict(self, query: str) -> IntentPrediction:
        key = self._cache_key(query)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        client = self._ensure_client()
        started = perf_counter()
        try:
            response = await client.post(
                f"{self.base_url}{SYSTEM_ONE_PATH}",
                json=self._payload(query),
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                timeout=self.timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise IntentProviderError("timeout") from exc
        except httpx.HTTPError as exc:
            # 连接错误、DNS、TLS 等。不记录 URL 之外的任何凭据信息。
            raise IntentProviderError(f"transport_{type(exc).__name__}") from exc

        if response.status_code != 200:
            # 401/422/429/529 都是文档列出的状态码；统一转成可审计的短标识。
            raise IntentProviderError(f"http_{response.status_code}")

        latency_ms = (perf_counter() - started) * 1000
        prediction = self._parse(response, latency_ms)
        self._record_usage(response, prediction.model or self.model)
        self._cache_put(key, prediction)
        return prediction

    def _record_usage(self, response: httpx.Response, model: str) -> None:
        if self._on_usage is None:
            return
        usage = response.json().get("usage")
        if not isinstance(usage, Mapping):
            return
        prompt = usage.get("input_tokens")
        completion = usage.get("output_tokens")
        if any(type(value) is not int or value < 0 for value in (prompt, completion)):
            return
        task = asyncio.create_task(self._report_usage(model, prompt, completion))
        self._usage_tasks.add(task)
        task.add_done_callback(self._usage_tasks.discard)

    async def _report_usage(self, model: str, prompt: int, completion: int) -> None:
        try:
            await self._on_usage(self._credential_id, "llm", model, prompt, completion)
        except Exception as exc:
            logger.warning("intent usage collection failed: {}", type(exc).__name__)

    def _parse(self, response: httpx.Response, latency_ms: float) -> IntentPrediction:
        try:
            body = response.json()
        except (json.JSONDecodeError, ValueError) as exc:
            raise IntentProviderError("invalid_json") from exc
        if not isinstance(body, Mapping):
            raise IntentProviderError("invalid_json")

        answers = body.get("answers")
        answer = answers.get(QUESTION_ID) if isinstance(answers, Mapping) else None
        if not isinstance(answer, Mapping):
            raise IntentProviderError("missing_answer")

        intent = intent_from_label(str(answer.get("choice", "")))
        if intent is None:
            raise IntentProviderError("unknown_choice")

        raw_probabilities = answer.get("probabilities")
        probabilities: dict[str, float] = {}
        if isinstance(raw_probabilities, Mapping):
            for label, value in raw_probabilities.items():
                try:
                    probabilities[str(label)] = float(value)
                except (TypeError, ValueError):
                    continue

        confidence = answer.get("confidence")
        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError):
            # 缺失 confidence 时退回该选项的概率；都没有就当 0，由仲裁器按
            # 低置信降级处理，而不是让一个未知量当成确定判定。
            confidence_value = probabilities.get(intent.value, 0.0)

        return IntentPrediction(
            intent=intent,
            confidence=confidence_value,
            probabilities=probabilities,
            model=str(body.get("model") or self.model),
            latency_ms=latency_ms,
        )


def build_typesafe_provider(
    *,
    api_key: str | None,
    base_url: str,
    model: str,
    timeout_seconds: float,
    cache_size: int,
) -> TypeSafeIntentProvider | None:
    """缺少 API key 时返回 ``None`` 并记录原因，不发起任何请求。"""
    if not (api_key or "").strip():
        logger.info(
            "TypeSafe intent provider disabled: no API key configured; "
            "intent classification runs in rules-only mode"
        )
        return None
    return TypeSafeIntentProvider(
        api_key=api_key or "",
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        cache_size=cache_size,
    )
