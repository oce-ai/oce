"""OpenAIReranker 单测：注入 mock httpx client 验 body shape 和过滤逻辑。"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from oce.infrastructure.embed.openai_reranker import OpenAIReranker
from oce.domain.services.search import SearchHit


@dataclass
class _Hit:
    """模拟 retrieval/store/vector_store.Hit 的最小字段（content + 可写 score）。"""
    content: str
    score: float = 0.0


def _fake_response(payload: dict, status_code: int = 200):
    """组个看起来像 httpx.Response 的对象，够 OpenAIReranker 用。"""
    resp = MagicMock()
    resp.json = MagicMock(return_value=payload)
    resp.raise_for_status = MagicMock(
        side_effect=None if 200 <= status_code < 400 else httpx.HTTPStatusError(
            "fake", request=MagicMock(), response=resp,
        )
    )
    resp.status_code = status_code
    return resp


def _make_reranker(*, top_n=10, min_score=0.1, response_payload=None,
                   raise_exc: Exception | None = None, instruct=None):
    """构造一个 OpenAIReranker，注入 mock httpx client。"""
    fake_client = MagicMock(spec=httpx.AsyncClient)
    if raise_exc:
        fake_client.post = AsyncMock(side_effect=raise_exc)
    else:
        fake_client.post = AsyncMock(return_value=_fake_response(response_payload or {}))

    reranker = OpenAIReranker(
        endpoint="https://dashscope.aliyuncs.com/compatible-api/v1/reranks",
        api_key="sk-xxx",
        model="qwen3-rerank",
        top_n=top_n,
        min_score=min_score,
        client=fake_client,
        instruct=instruct,
    )
    return reranker, fake_client


# ── 边界 ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rerank_empty_hits_returns_empty():
    reranker, client = _make_reranker(response_payload={"results": []})
    result = await reranker.rerank("q", [])
    assert result == []
    client.post.assert_not_awaited()  # 空就不发请求


# ── SiliconFlow body shape ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rerank_body_shape_disables_return_documents():
    reranker, client = _make_reranker(
        response_payload={"results": [{"index": 0, "relevance_score": 0.9}]},
    )
    await reranker.rerank("q", [_Hit(content="d1")])

    body = client.post.call_args.kwargs["json"]
    assert body["model"] == "qwen3-rerank"
    assert body["query"] == "q"
    assert body["documents"] == ["d1"]
    assert body["return_documents"] is False


@pytest.mark.asyncio
async def test_rerank_documents_include_source_location_when_available():
    reranker, client = _make_reranker(
        response_payload={"results": [{"index": 0, "relevance_score": 0.9}]},
    )
    hit = SearchHit(
        blob_name="a" * 64,
        path="src/auth.py",
        content="def authorize(): pass",
        score=0.4,
        start_line=12,
        end_line=12,
    )

    await reranker.rerank("authorization", [hit])

    body = client.post.call_args.kwargs["json"]
    assert body["documents"] == [
        "File: src/auth.py\nLines: 12-12\n\ndef authorize(): pass"
    ]


@pytest.mark.asyncio
async def test_rerank_body_shape_omits_instruction_when_none():
    reranker, client = _make_reranker(response_payload={"results": []})
    await reranker.rerank("q", [_Hit(content="d1")])
    body = client.post.call_args.kwargs["json"]
    assert "instruction" not in body


@pytest.mark.asyncio
async def test_rerank_body_shape_includes_instruction_when_set():
    reranker, client = _make_reranker(
        response_payload={"results": []},
        instruct="Given a web search query, retrieve relevant passages.",
    )
    await reranker.rerank("q", [_Hit(content="d1")])
    body = client.post.call_args.kwargs["json"]
    assert body["instruction"] == "Given a web search query, retrieve relevant passages."


@pytest.mark.asyncio
async def test_rerank_authorization_header():
    reranker, client = _make_reranker(response_payload={"results": []})
    await reranker.rerank("q", [_Hit(content="d1")])
    headers = client.post.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer sk-xxx"
    assert headers["Content-Type"] == "application/json"


# ── 排序 / 过滤 ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rerank_filters_below_min_score():
    """relevance_score < min_score 的丢弃。"""
    reranker, _ = _make_reranker(
        min_score=0.5,
        response_payload={"results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.3},   # < 0.5 应被丢
            {"index": 2, "relevance_score": 0.6},
        ]},
    )
    hits = [_Hit(content="a"), _Hit(content="b"), _Hit(content="c")]
    result = await reranker.rerank("q", hits)
    assert [h.content for h in result] == ["a", "c"]
    assert result[0].score == pytest.approx(0.9)
    assert result[1].score == pytest.approx(0.6)


@pytest.mark.asyncio
async def test_rerank_truncates_to_top_n():
    """API 多返了，按 top_n 截断。"""
    reranker, _ = _make_reranker(
        top_n=2,
        response_payload={"results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 1, "relevance_score": 0.8},
            {"index": 2, "relevance_score": 0.7},
        ]},
    )
    hits = [_Hit(content="a"), _Hit(content="b"), _Hit(content="c")]
    result = await reranker.rerank("q", hits)
    assert len(result) == 2
    assert [h.content for h in result] == ["a", "b"]


@pytest.mark.asyncio
async def test_rerank_top_n_in_body_caps_at_documents_size():
    """top_n 不会超过 documents 数（避免 dashscope 报参数错）。"""
    reranker, client = _make_reranker(
        top_n=100,
        response_payload={"results": []},
    )
    hits = [_Hit(content="a"), _Hit(content="b"), _Hit(content="c")]
    await reranker.rerank("q", hits)
    body = client.post.call_args.kwargs["json"]
    assert body["top_n"] == 3, "top_n 应该按 hits 数压低"


# ── 失败降级 ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rerank_http_failure_returns_original_order():
    """HTTP 报错时不抛异常，返回原顺序前 top_n 条。"""
    reranker, _ = _make_reranker(
        top_n=2,
        raise_exc=httpx.ConnectError("network down"),
    )
    hits = [_Hit(content="a"), _Hit(content="b"), _Hit(content="c")]
    result = await reranker.rerank("q", hits)
    assert [h.content for h in result] == ["a", "b"]


# ── on_usage 上报：成功路径触发 fire-and-forget 回调 ───────────────────────
@pytest.mark.asyncio
async def test_rerank_on_usage_callback_invoked_on_success():
    """rerank 成功后调用 on_usage(cred_id, 'rerank', tokens, latency_ms)；usage 缺字段时 tokens=0。"""
    import asyncio as _asyncio

    captured: list[tuple] = []

    async def _on_usage(cid, kind, tokens, latency_ms):
        captured.append((cid, kind, tokens, latency_ms))

    fake_client = MagicMock(spec=httpx.AsyncClient)
    fake_client.post = AsyncMock(return_value=_fake_response({
        "results": [{"index": 0, "relevance_score": 0.9}],
        "meta": {"tokens": {"input_tokens": 40, "output_tokens": 2}},
    }))
    rk = OpenAIReranker(
        endpoint="https://x/rerank", api_key="sk", model="qwen3-rerank",
        client=fake_client, credential_id=7, on_usage=_on_usage,
    )
    await rk.rerank("q", [_Hit(content="d1")])
    # fire-and-forget：等一轮事件循环让 task 执行
    await _asyncio.sleep(0)
    await _asyncio.sleep(0)
    assert captured, "on_usage 应至少被调用一次"
    cid, kind, tokens, latency = captured[0]
    assert (cid, kind, tokens) == (7, "rerank", 42)
    assert latency >= 0


@pytest.mark.asyncio
async def test_rerank_on_usage_not_invoked_on_http_failure():
    """HTTP 失败 → 直接 return hits[:top_n]，不调用 on_usage（无需计费）。"""
    import asyncio as _asyncio
    captured: list[tuple] = []

    async def _on_usage(cid, kind, tokens, latency_ms):
        captured.append((cid, kind, tokens, latency_ms))

    fake_client = MagicMock(spec=httpx.AsyncClient)
    fake_client.post = AsyncMock(side_effect=httpx.ConnectError("down"))
    rk = OpenAIReranker(
        endpoint="https://x/rerank", api_key="sk", model="qwen3-rerank",
        client=fake_client, credential_id=7, on_usage=_on_usage,
    )
    await rk.rerank("q", [_Hit(content="d1")])
    await _asyncio.sleep(0)
    assert captured == [], "失败路径不应该触发 usage 上报"
