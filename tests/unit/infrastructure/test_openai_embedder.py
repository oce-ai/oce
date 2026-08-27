"""OpenAI-compatible embedding input budgeting tests."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from oce.infrastructure.embed.openai_embedder import OpenAIEmbedder


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def create(self, **kwargs):
        texts = list(kwargs["input"])
        self.calls.append(texts)
        data = []
        for index, text in enumerate(texts):
            vector = [1.0, 0.0] if text[0] < "f" else [0.0, 1.0]
            data.append(SimpleNamespace(index=index, embedding=vector))
        return SimpleNamespace(
            data=data,
            usage=SimpleNamespace(total_tokens=sum(map(len, texts))),
        )


class _FakeClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddings()

    async def close(self) -> None:
        return None


def _make_embedder(**kwargs) -> tuple[OpenAIEmbedder, _FakeClient]:
    client = _FakeClient()
    embedder = OpenAIEmbedder(
        client,
        "test-model",
        2,
        max_batch_size=kwargs.get("max_batch_size", 32),
        max_concurrency=1,
        max_batch_chars=kwargs.get("max_batch_chars", 32_000),
        max_input_chars=kwargs.get("max_input_chars", 8_000),
        input_overlap_chars=kwargs.get("input_overlap_chars", 0),
    )
    return embedder, client


@pytest.mark.asyncio
async def test_batching_respects_item_and_total_character_limits():
    embedder, client = _make_embedder(
        max_batch_size=3,
        max_batch_chars=6,
        max_input_chars=6,
    )

    await embedder.embed_documents(["aaaa", "bbbb", "cc"])

    assert client.embeddings.calls == [["aaaa"], ["bbbb", "cc"]]


@pytest.mark.asyncio
async def test_long_input_is_split_and_pooled_to_one_normalized_vector():
    embedder, client = _make_embedder(
        max_batch_chars=6,
        max_input_chars=5,
    )

    vectors = await embedder.embed_documents(["abcdefghij"])

    assert client.embeddings.calls == [["abcde"], ["fghij"]]
    assert len(vectors) == 1
    assert vectors[0] == pytest.approx([1 / math.sqrt(2), 1 / math.sqrt(2)])


@pytest.mark.asyncio
async def test_short_input_preserves_provider_vector():
    embedder, _ = _make_embedder()

    assert await embedder.embed_query("abc") == [1.0, 0.0]


def test_invalid_input_budget_is_rejected():
    client = _FakeClient()

    with pytest.raises(ValueError, match="character budgets"):
        OpenAIEmbedder(
            client,
            "test-model",
            2,
            max_batch_chars=4,
            max_input_chars=5,
        )
