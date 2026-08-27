"""SiliconFlow/Cohere 风格的异步 rerank 客户端。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Awaitable, Callable

import httpx
from loguru import logger

UsageCallback = Callable[[int, str, int, int], Awaitable[None]]


class OpenAIReranker:
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        model: str,
        top_n: int = 10,
        min_score: float = 0.05,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        instruct: str | None = None,
        credential_id: int = 0,
        on_usage: UsageCallback | None = None,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._top_n = top_n
        self._min_score = min_score
        self._instruct = instruct
        self._credential_id = credential_id
        self._on_usage = on_usage
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def rerank(self, query: str, hits: list[Any]) -> list[Any]:
        if not hits:
            return []
        body: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": [self._document_text(hit) for hit in hits],
            "top_n": min(self._top_n, len(hits)),
            "return_documents": False,
        }
        if self._instruct:
            body["instruction"] = self._instruct
        try:
            response = await self._client.post(
                self._endpoint,
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Rerank request failed; using retrieval order: {}", exc)
            return hits[: self._top_n]

        ranked: list[tuple[int, float]] = []
        for item in payload.get("results", []):
            index = item.get("index")
            score = item.get("relevance_score", item.get("score", 0.0))
            if isinstance(index, int) and 0 <= index < len(hits) and score >= self._min_score:
                ranked.append((index, float(score)))
        ranked.sort(key=lambda pair: pair[1], reverse=True)

        output: list[Any] = []
        for index, score in ranked[: self._top_n]:
            hit = hits[index]
            try:
                hit = replace(hit, score=score)
            except TypeError:
                try:
                    hit.score = score
                except (AttributeError, TypeError):
                    pass
            output.append(hit)

        if self._on_usage is not None:
            meta = payload.get("meta") or {}
            token_meta = meta.get("tokens") or {}
            tokens = sum(
                int(token_meta.get(key, 0) or 0)
                for key in ("input_tokens", "output_tokens", "image_tokens")
            )
            await self._on_usage(
                self._credential_id,
                "rerank",
                tokens,
                0,
            )
        return output

    @staticmethod
    def _document_text(hit: Any) -> str:
        path = getattr(hit, "path", "")
        start_line = getattr(hit, "start_line", None)
        end_line = getattr(hit, "end_line", None)
        if path and isinstance(start_line, int) and isinstance(end_line, int):
            return (
                f"File: {path}\nLines: {start_line}-{end_line}\n\n"
                f"{hit.content}"
            )
        return hit.content

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
