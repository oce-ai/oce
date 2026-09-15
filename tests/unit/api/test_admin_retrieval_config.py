"""/admin/bench/retrieval-config 热改端点契约测试。

仿 test_admin_queue.py：dependency_overrides + ASGITransport + mock admin auth。
覆盖 409 闸（未开热改）、GET/POST 契约、422（未知 key / 越界）、鉴权。
reconfigurator 本身的校验逻辑在 test_reconfigure.py 里逐条测；此处只测路由的映射与闸。
"""

from __future__ import annotations

import httpx
import pytest
from fastapi import Header

from oce.api.admin_router import hot_config_allowed
from oce.api.router import get_application
from oce.application.commands.reconfigure import (
    HOT_CONFIG_ENV,
    HOT_RETRIEVAL_FIELDS,
    HotConfigError,
    ReconfigureResult,
    RetrievalConfigResult,
)
from oce.auth import _unauthorized, verify_admin_key
from oce.main import app


async def _mock_admin_auth(authorization: str | None = Header(default=None)) -> str:
    if authorization is None or not authorization.startswith("Bearer "):
        raise _unauthorized("missing admin key")
    return authorization.removeprefix("Bearer ")


_EFFECTIVE = {
    "retrieval": {"default_top_k": 50, "rrf_k": 60},
    "flags": {"rerank_enabled": True, "llm_rerank_enabled": False},
    "milvus": {"hnsw_ef_search": 512},
    "rerank": {"top_n": 10, "min_score": 0.05},
}


class StubBenchApp:
    """门面替身：retrieval_config 读快照；reconfigure_retrieval 做最小 key 校验。"""

    def __init__(self) -> None:
        self.generation = 0

    async def retrieval_config(self) -> RetrievalConfigResult:
        return RetrievalConfigResult(generation=self.generation, effective=_EFFECTIVE)

    async def reconfigure_retrieval(
        self, *, retrieval_patch=None, flags=None, milvus_patch=None, rerank_patch=None
    ) -> ReconfigureResult:
        # 模拟 reconfigurator 的白名单校验，验证路由把 HotConfigError 映射成 422
        unknown = sorted(set(retrieval_patch or {}) - set(HOT_RETRIEVAL_FIELDS))
        if unknown:
            raise HotConfigError(
                f"unknown retrieval field(s): {', '.join(unknown)}",
                code="HOT_CONFIG_UNKNOWN_FIELD",
                details={"unknown": unknown},
            )
        if (retrieval_patch or {}).get("default_top_k", 50) > 200:
            raise HotConfigError(
                "retrieval patch failed validation",
                code="HOT_CONFIG_OUT_OF_RANGE",
                details={"group": "retrieval"},
            )
        self.generation += 1
        new_eff = dict(_EFFECTIVE)
        new_eff["retrieval"] = {**_EFFECTIVE["retrieval"], **(retrieval_patch or {})}
        return ReconfigureResult(
            generation=self.generation,
            effective=new_eff,
            reranker_reloaded=bool(rerank_patch) or "rerank_enabled" in (flags or {}),
        )


def _client(*, allowed: bool = True, stub: StubBenchApp | None = None) -> httpx.AsyncClient:
    # 传入共享 stub 可跨请求保留 generation（测 read-after-write 持久化）；
    # 默认每请求一个新实例（各用例彼此隔离）。
    shared = stub if stub is not None else StubBenchApp()
    app.dependency_overrides[get_application] = lambda: shared
    app.dependency_overrides[verify_admin_key] = _mock_admin_auth
    app.dependency_overrides[hot_config_allowed] = lambda: allowed
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


_AUTH = {"Authorization": "Bearer sk-admin"}


# ---------------------------------------------------------------------------
# GET 契约
# ---------------------------------------------------------------------------


async def test_get_returns_generation_and_effective():
    async with _client() as client:
        response = await client.get("/admin/bench/retrieval-config", headers=_AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["generation"] == 0
    assert body["effective"]["retrieval"]["default_top_k"] == 50
    assert body["effective"]["milvus"]["hnsw_ef_search"] == 512
    # GET 不带 reranker_reloaded（恒为 None）
    assert body["reranker_reloaded"] is None


async def test_get_effective_excludes_secrets():
    async with _client() as client:
        response = await client.get("/admin/bench/retrieval-config", headers=_AUTH)
    assert "api_key" not in response.json()["effective"]["rerank"]


# ---------------------------------------------------------------------------
# POST 契约
# ---------------------------------------------------------------------------


async def test_post_applies_patch_and_increments_generation():
    async with _client() as client:
        response = await client.post(
            "/admin/bench/retrieval-config",
            headers=_AUTH,
            json={"retrieval": {"default_top_k": 30}},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["generation"] == 1
    assert body["effective"]["retrieval"]["default_top_k"] == 30
    assert body["reranker_reloaded"] is False


async def test_post_rerank_patch_reports_reload():
    async with _client() as client:
        response = await client.post(
            "/admin/bench/retrieval-config",
            headers=_AUTH,
            json={"rerank": {"top_n": 5}},
        )
    assert response.status_code == 200
    assert response.json()["reranker_reloaded"] is True


async def test_post_read_after_write_sequence():
    """harness 的核心用法：POST 改 → GET 确认 generation 前进 + effective 含新值。

    用共享 stub 模拟进程内持久状态：POST 与 GET 命中同一 reconfigurator，故 GET 必须
    看到 POST 推进的 generation 和生效值——这正是 harness「改完 GET 一次确认才开跑」的契约。
    """
    stub = StubBenchApp()
    async with _client(stub=stub) as client:
        get0 = await client.get("/admin/bench/retrieval-config", headers=_AUTH)
        assert get0.json()["generation"] == 0

        post = await client.post(
            "/admin/bench/retrieval-config",
            headers=_AUTH,
            json={"retrieval": {"default_top_k": 77}},
        )
        assert post.json()["generation"] == 1

        get1 = await client.get("/admin/bench/retrieval-config", headers=_AUTH)
    # read-after-write：generation 单调前进，effective 反映已下发的 patch
    assert get1.json()["generation"] == 1
    assert post.json()["effective"]["retrieval"]["default_top_k"] == 77


# ---------------------------------------------------------------------------
# 422：未知 key / 越界
# ---------------------------------------------------------------------------


async def test_post_unknown_key_returns_422():
    async with _client() as client:
        response = await client.post(
            "/admin/bench/retrieval-config",
            headers=_AUTH,
            json={"retrieval": {"default_topkk": 5}},
        )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "HOT_CONFIG_UNKNOWN_FIELD"
    assert "default_topkk" in detail["unknown"]


async def test_post_out_of_range_returns_422():
    async with _client() as client:
        response = await client.post(
            "/admin/bench/retrieval-config",
            headers=_AUTH,
            json={"retrieval": {"default_top_k": 99999}},
        )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "HOT_CONFIG_OUT_OF_RANGE"


# ---------------------------------------------------------------------------
# 409 闸：未开热改
# ---------------------------------------------------------------------------


async def test_get_forbidden_when_hot_config_disabled():
    async with _client(allowed=False) as client:
        response = await client.get("/admin/bench/retrieval-config", headers=_AUTH)
    assert response.status_code == 409
    assert HOT_CONFIG_ENV in response.json()["detail"]


async def test_post_forbidden_when_hot_config_disabled():
    async with _client(allowed=False) as client:
        response = await client.post(
            "/admin/bench/retrieval-config",
            headers=_AUTH,
            json={"retrieval": {"default_top_k": 30}},
        )
    assert response.status_code == 409


async def test_gate_dependency_reads_env(monkeypatch):
    """真实 hot_config_allowed()：仅 OCE_BENCH_HOT_CONFIG=allow 放行（大小写/空白容错）。"""
    app.dependency_overrides.pop(hot_config_allowed, None)
    monkeypatch.delenv(HOT_CONFIG_ENV, raising=False)
    assert hot_config_allowed() is False
    monkeypatch.setenv(HOT_CONFIG_ENV, "nope")
    assert hot_config_allowed() is False
    monkeypatch.setenv(HOT_CONFIG_ENV, "allow")
    assert hot_config_allowed() is True
    monkeypatch.setenv(HOT_CONFIG_ENV, "  ALLOW  ")
    assert hot_config_allowed() is True


# ---------------------------------------------------------------------------
# 鉴权
# ---------------------------------------------------------------------------


async def test_requires_admin_auth():
    async with _client() as client:
        response = await client.get("/admin/bench/retrieval-config")
    assert response.status_code == 401
