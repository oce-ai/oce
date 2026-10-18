"""SiliconFlow/Cohere 风格的异步 rerank 客户端。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Awaitable, Callable

import httpx
from loguru import logger

# 用量回调：(credential_id, kind, model, prompt_tokens, completion_tokens)
UsageCallback = Callable[[int, str, str, int, int], Awaitable[None]]


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
        char_budget: int = 32000,
    ) -> None:
        self._endpoint = endpoint
        self._api_key = api_key
        self._model = model
        self._top_n = top_n
        self._min_score = min_score
        self._char_budget = char_budget
        self._instruct = instruct
        self._credential_id = credential_id
        self._on_usage = on_usage
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def rerank(self, query: str, hits: list[Any]) -> list[Any]:
        if not hits:
            return []
        documents, keep_idx = self._slim_documents(hits)
        if not documents:
            return hits[: self._top_n]
        body: dict[str, Any] = {
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": min(self._top_n, len(documents)),
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
            if isinstance(index, int) and 0 <= index < len(keep_idx) and score >= self._min_score:
                ranked.append((keep_idx[index], float(score)))
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
            # rerank 无 prompt/completion 之分：总量记入 prompt，completion=0
            await self._on_usage(
                self._credential_id,
                "rerank",
                self._model,
                tokens,
                0,
            )
        return output

    def _slim_documents(self, hits: list[Any]) -> tuple[list[str], list[int]]:
        """按总字符预算削减发送候选，返回 (发送文本列表, 对应的原始下标)。

        远程 n_ctx=32768 tokens；CJK 最坏 ~0.76 tok/char，32k 字符 ≈ 24.3k
        token，留足 query/模板余量后不会溢出（与 rerank_wrap._slim_docs 对齐）。
        先整篇收，预算不够截当前篇，丢弃后续。index 必须回带：远端
        results[].index 是相对【发送列表】的，调用方要拿它取原始 hit 对象，
        不映射会静默取错文档。
        """
        documents: list[str] = []
        keep_idx: list[int] = []
        budget = self._char_budget
        for i, hit in enumerate(hits):
            if budget <= 0:
                break
            text = self._document_text(hit)
            if len(text) > budget:
                text = text[:budget]
            documents.append(text)
            keep_idx.append(i)
            budget -= len(text)
        if len(documents) < len(hits):
            logger.warning(
                "Rerank char budget {} exceeded; sending {}/{} candidates",
                self._char_budget,
                len(documents),
                len(hits),
            )
        return documents, keep_idx

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
