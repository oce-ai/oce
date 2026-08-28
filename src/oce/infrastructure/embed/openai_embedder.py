"""OpenAI 兼容的异步 embedding 客户端。"""

from __future__ import annotations

import asyncio
import math
from typing import Awaitable, Callable

import httpx
from openai import AsyncOpenAI

# 用量回调：(credential_id, kind, model, prompt_tokens, completion_tokens)
UsageCallback = Callable[[int, str, str, int, int], Awaitable[None]]


class OpenAIEmbedder:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        dimensions: int,
        *,
        max_batch_size: int = 32,
        max_concurrency: int = 4,
        max_batch_chars: int = 32_000,
        max_input_chars: int = 8_000,
        input_overlap_chars: int = 400,
        credential_id: int = 0,
        on_usage: UsageCallback | None = None,
        query_instruction: str = "",
    ) -> None:
        if max_batch_size < 1 or max_concurrency < 1:
            raise ValueError("Embedding batch size and concurrency must be positive")
        if max_input_chars < 1 or max_batch_chars < max_input_chars:
            raise ValueError("Embedding character budgets are invalid")
        if input_overlap_chars < 0 or input_overlap_chars >= max_input_chars:
            raise ValueError("Embedding input overlap must be smaller than its window")
        self._client = client
        self._model = model
        self._dimensions = dimensions
        self._max_batch_size = max_batch_size
        self._max_concurrency = max_concurrency
        self._max_batch_chars = max_batch_chars
        self._max_input_chars = max_input_chars
        self._input_overlap_chars = input_overlap_chars
        self._credential_id = credential_id
        self._on_usage = on_usage
        self._query_instruction = query_instruction

    @classmethod
    def from_endpoint(
        cls,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        dimensions: int,
        max_batch_size: int = 32,
        max_concurrency: int = 4,
        max_batch_chars: int = 32_000,
        max_input_chars: int = 8_000,
        input_overlap_chars: int = 400,
        timeout: float = 60.0,
        credential_id: int = 0,
        on_usage: UsageCallback | None = None,
        proxy: str | None = None,
        query_instruction: str = "",
        **_: object,
    ) -> "OpenAIEmbedder":
        base_url = endpoint.rstrip("/")
        if base_url.endswith("/embeddings"):
            base_url = base_url[: -len("/embeddings")]
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=max_concurrency * 2),
            proxy=proxy,
        )
        client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            http_client=http_client,
        )
        return cls(
            client,
            model,
            dimensions,
            max_batch_size=max_batch_size,
            max_concurrency=max_concurrency,
            max_batch_chars=max_batch_chars,
            max_input_chars=max_input_chars,
            input_overlap_chars=input_overlap_chars,
            credential_id=credential_id,
            on_usage=on_usage,
            query_instruction=query_instruction,
        )

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        segment_groups = [self._split_input(text) for text in texts]
        segments = [segment for group in segment_groups for segment in group]
        segment_vectors = await self._embed_segments(segments)

        vectors: list[list[float]] = []
        offset = 0
        for group in segment_groups:
            next_offset = offset + len(group)
            vectors.append(
                self._pool_vectors(segment_vectors[offset:next_offset], group)
            )
            offset = next_offset
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        # 添加 query instruction（如果配置了）
        if self._query_instruction:
            text = self._query_instruction + text
        return (await self.embed_documents([text]))[0]

    async def _embed_segments(self, texts: list[str]) -> list[list[float]]:
        batches = self._make_batches(texts)
        semaphore = asyncio.Semaphore(self._max_concurrency)

        async def run(batch: list[str]) -> list[list[float]]:
            async with semaphore:
                return await self._embed_batch(batch)

        results = await asyncio.gather(*(run(batch) for batch in batches))
        return [vector for batch in results for vector in batch]

    def _make_batches(self, texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = []
        batch: list[str] = []
        batch_chars = 0
        for text in texts:
            text_chars = max(1, len(text))
            if batch and (
                len(batch) >= self._max_batch_size
                or batch_chars + text_chars > self._max_batch_chars
            ):
                batches.append(batch)
                batch = []
                batch_chars = 0
            batch.append(text)
            batch_chars += text_chars
        if batch:
            batches.append(batch)
        return batches

    def _split_input(self, text: str) -> list[str]:
        if len(text) <= self._max_input_chars:
            return [text]

        segments: list[str] = []
        start = 0
        while start < len(text):
            hard_end = min(start + self._max_input_chars, len(text))
            end = hard_end
            if hard_end < len(text):
                search_start = start + self._max_input_chars // 2
                boundary = max(
                    text.rfind("\n", search_start, hard_end),
                    text.rfind(" ", search_start, hard_end),
                )
                if boundary >= search_start:
                    end = boundary + 1
            segments.append(text[start:end])
            if end >= len(text):
                break
            next_start = max(start + 1, end - self._input_overlap_chars)
            start = next_start
        return segments

    def _pool_vectors(
        self,
        vectors: list[list[float]],
        segments: list[str],
    ) -> list[float]:
        if len(vectors) != len(segments) or not vectors:
            raise RuntimeError("Embedding segment count mismatch")
        if len(vectors) == 1:
            return vectors[0]

        weights = [max(1, len(segment)) for segment in segments]
        total_weight = sum(weights)
        pooled = [
            sum(vector[index] * weight for vector, weight in zip(vectors, weights))
            / total_weight
            for index in range(self._dimensions)
        ]
        norm = math.sqrt(sum(value * value for value in pooled))
        return [value / norm for value in pooled] if norm else pooled

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimensions,
            encoding_format="float",
        )
        data = sorted(response.data, key=lambda item: item.index)
        vectors = [list(item.embedding) for item in data]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Embedding response count mismatch: expected {len(texts)}, got {len(vectors)}"
            )
        if any(len(vector) != self._dimensions for vector in vectors):
            raise RuntimeError("Embedding response dimension mismatch")
        if self._on_usage is not None:
            tokens = int(getattr(response.usage, "total_tokens", 0) or 0)
            # embed 无 prompt/completion 之分：总量记入 prompt，completion=0
            await self._on_usage(self._credential_id, "embed", self._model, tokens, 0)
        return vectors

    async def close(self) -> None:
        await self._client.close()
