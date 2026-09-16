"""评测客户端传输层测试（httpx.MockTransport 假服务，不起真进程）。

覆盖：上传 poison-split 二分重试、find-missing / blob-status 的**字段名差异**
（unknown_memory_names vs unknown_blob_names）、嵌入轮询（进度/超时/unknown 立即失败）、
检索耗时双视角、以及 L0 热改的 **read-after-write 验证**（generation 前进 + effective ⊇
patch）—— 后者是热调参可信度的基石，逐条对应 client.reconfigure 的断言。
"""

from __future__ import annotations

import json

import httpx
import pytest

from oce.bench.blobs import SourceBlob
from oce.bench.client import (
    BenchClient,
    BenchHTTPError,
    ReconfigureRejected,
)


class FakeService:
    """按 path 分发的手写假服务；记录请求、可编程响应。"""

    def __init__(self) -> None:
        self.uploaded: list[dict] = []
        self.config_generation = 0
        self.effective = {
            "retrieval": {"default_top_k": 50, "rrf_k": 60},
            "flags": {"rerank_enabled": True, "llm_rerank_enabled": False},
            "milvus": {"hnsw_ef_search": 512},
            "rerank": {"top_n": 10, "min_score": 0.05},
        }
        # 可编程故障注入
        self.upload_fail_paths: set[str] = set()
        self.status_sequence: list[tuple[list[str], list[str]]] = []
        self.retrieve_response = {
            "formatted_retrieval": "Path: a.py\nsnippet",
            "codebase_retrieval_elapsed_ms": 42,
        }
        self.config_post_status = 200
        self.config_post_detail: dict | str = {}
        self.upload_status: int | None = None
        self.auth_headers: list[tuple[str, str | None]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}
        self.auth_headers.append((path, request.headers.get("authorization")))

        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})

        if path == "/batch-upload":
            if self.upload_status is not None:
                return httpx.Response(self.upload_status, text="upload failed")
            blobs = body.get("blobs", [])
            paths = [b["path"] for b in blobs]
            # 若本批含"毒文件"，整批 422（触发 poison-split 二分）
            if any(p in self.upload_fail_paths for p in paths):
                return httpx.Response(422, text="poisoned batch")
            names = [f"name-{p}" for p in paths]
            self.uploaded.extend(blobs)
            return httpx.Response(200, json={"blob_names": names})

        if path == "/find-missing":
            return httpx.Response(
                200,
                json={"unknown_memory_names": [], "nonindexed_blob_names": []},
            )

        if path == "/agents/blob-status":
            if self.status_sequence:
                unknown, nonindexed = self.status_sequence.pop(0)
            else:
                unknown, nonindexed = [], []
            return httpx.Response(
                200,
                json={
                    "unknown_blob_names": unknown,
                    "nonindexed_blob_names": nonindexed,
                    "checkpoint_not_found": False,
                },
            )

        if path == "/agents/codebase-retrieval":
            return httpx.Response(200, json=self.retrieve_response)

        if path == "/admin/bench/retrieval-config":
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "generation": self.config_generation,
                        "effective": self.effective,
                        "reranker_reloaded": None,
                    },
                )
            # POST：应用 patch（模拟 reconfigurator）
            if self.config_post_status != 200:
                return httpx.Response(
                    self.config_post_status, json={"detail": self.config_post_detail}
                )
            patch = body
            self.config_generation += 1
            for group in ("retrieval", "flags", "milvus", "rerank"):
                for key, value in (patch.get(group) or {}).items():
                    # 模拟 pydantic 强转：字符串 -> 目标类型
                    self.effective[group][key] = _coerce_like_server(value)
            return httpx.Response(
                200,
                json={
                    "generation": self.config_generation,
                    "effective": self.effective,
                    "reranker_reloaded": bool(patch.get("rerank")),
                },
            )

        return httpx.Response(404, text=f"no route {path}")


def _coerce_like_server(value):
    """模拟服务端 pydantic 对字符串的强转（"30"->30, "false"->False）。"""
    if not isinstance(value, str):
        return value
    low = value.strip().lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _client(service: FakeService) -> BenchClient:
    return BenchClient(
        "http://bench.test",
        "sk-test",
        transport=httpx.MockTransport(service.handler),
        poll_interval=0.0,  # 轮询测试不真 sleep
    )


# ---------------------------------------------------------------------------
# health / 上传
# ---------------------------------------------------------------------------


async def test_health_no_auth_needed():
    service = FakeService()
    async with _client(service) as client:
        assert (await client.health())["status"] == "ok"


async def test_upload_batch_returns_names():
    service = FakeService()
    async with _client(service) as client:
        names, skipped = await client.upload_batch(
            [SourceBlob("a.py", "x"), SourceBlob("b.py", "y")]
        )
    assert names == ["name-a.py", "name-b.py"]
    assert skipped == []


async def test_upload_poison_split_recovers_good_files():
    """一个毒文件让整批 422 -> 二分把好文件捞回、只把坏文件记 skipped。"""
    service = FakeService()
    service.upload_fail_paths = {"bad.py"}
    async with _client(service) as client:
        names, skipped = await client.upload_batch(
            [SourceBlob("a.py", "x"), SourceBlob("bad.py", "y"), SourceBlob("c.py", "z")]
        )
    assert sorted(names) == ["name-a.py", "name-c.py"]
    assert len(skipped) == 1
    assert "bad.py" in skipped[0]
    assert "422" in skipped[0]


async def test_upload_http_error_on_non_poison_single():
    """单文件批返回非 200 -> 记 skipped（不再二分）。"""
    service = FakeService()
    service.upload_fail_paths = {"only.py"}
    async with _client(service) as client:
        names, skipped = await client.upload_batch([SourceBlob("only.py", "x")])
    assert names == []
    assert len(skipped) == 1


@pytest.mark.parametrize("status", [401, 403, 429, 500, 503])
async def test_upload_systemic_error_aborts_without_poison_split(status):
    service = FakeService()
    service.upload_status = status
    async with _client(service) as client:
        with pytest.raises(BenchHTTPError) as exc:
            await client.upload_batch([SourceBlob("a.py", "x"), SourceBlob("b.py", "y")])
    assert exc.value.status_code == status
    assert sum(path == "/batch-upload" for path, _ in service.auth_headers) == 1


async def test_data_and_admin_endpoints_use_separate_keys():
    service = FakeService()
    client = BenchClient(
        "http://bench.test",
        "data-key",
        admin_api_key="admin-key",
        transport=httpx.MockTransport(service.handler),
    )
    async with client:
        await client.upload_batch([SourceBlob("a.py", "x")])
        await client.get_config()
    headers = dict(service.auth_headers)
    assert headers["/batch-upload"] == "Bearer data-key"
    assert headers["/admin/bench/retrieval-config"] == "Bearer admin-key"


# ---------------------------------------------------------------------------
# find-missing / blob-status 字段名差异
# ---------------------------------------------------------------------------


async def test_find_missing_uses_memory_names_field():
    service = FakeService()
    async with _client(service) as client:
        unknown, nonindexed = await client.find_missing(["n1", "n2"])
    assert unknown == [] and nonindexed == []


async def test_blob_status_field_names():
    service = FakeService()
    service.status_sequence = [(["u1"], ["p1", "p2"])]
    async with _client(service) as client:
        unknown, nonindexed = await client.blob_status(["n1"])
    # blob-status 用 unknown_blob_names（与 find-missing 的 unknown_memory_names 不同）
    assert unknown == ["u1"]
    assert nonindexed == ["p1", "p2"]


# ---------------------------------------------------------------------------
# 嵌入轮询
# ---------------------------------------------------------------------------


async def test_wait_for_embedding_polls_until_indexed():
    service = FakeService()
    # 前两次仍有 pending，第三次清空 -> 完成
    service.status_sequence = [([], ["p1", "p2"]), ([], ["p1"]), ([], [])]
    async with _client(service) as client:
        await client.wait_for_embedding(["n1", "n2", "n3"], timeout=10)
    assert service.status_sequence == []  # 消费完


async def test_wait_for_embedding_empty_is_noop():
    service = FakeService()
    async with _client(service) as client:
        await client.wait_for_embedding([], timeout=1)  # 不发请求


async def test_wait_for_embedding_unknown_raises():
    service = FakeService()
    service.status_sequence = [(["ghost"], [])]
    async with _client(service) as client:
        with pytest.raises(RuntimeError, match="unknown names"):
            await client.wait_for_embedding(["n1"], timeout=10)


async def test_wait_for_embedding_timeout():
    service = FakeService()
    # 永远有 pending -> 超时（poll_interval=0 让循环快转，靠 timeout 退出）
    service.status_sequence = [([], ["p1"])] * 10000
    async with _client(service) as client:
        with pytest.raises(TimeoutError, match="did not complete"):
            await client.wait_for_embedding(["n1", "n2"], timeout=0.05)


# ---------------------------------------------------------------------------
# 检索
# ---------------------------------------------------------------------------


async def test_retrieve_returns_both_latencies():
    service = FakeService()
    async with _client(service) as client:
        outcome = await client.retrieve("how does X work", added_blobs=["n1"])
    assert outcome.formatted.startswith("Path: a.py")
    assert outcome.server_elapsed_ms == 42
    assert outcome.client_elapsed_ms >= 0


async def test_retrieve_http_error_raises():
    service = FakeService()

    def failing(request):
        if request.url.path == "/agents/codebase-retrieval":
            return httpx.Response(500, text="boom")
        return service.handler(request)

    client = BenchClient(
        "http://bench.test", "sk", transport=httpx.MockTransport(failing)
    )
    async with client:
        with pytest.raises(BenchHTTPError) as ei:
            await client.retrieve("q", added_blobs=["n1"])
    assert ei.value.status_code == 500


# ---------------------------------------------------------------------------
# L0 热改 read-after-write
# ---------------------------------------------------------------------------


async def test_get_config():
    service = FakeService()
    async with _client(service) as client:
        snap = await client.get_config()
    assert snap.generation == 0
    assert snap.effective["retrieval"]["default_top_k"] == 50


async def test_reconfigure_read_after_write_succeeds():
    service = FakeService()
    async with _client(service) as client:
        ack = await client.reconfigure(retrieval={"default_top_k": "30"})
    assert ack.generation == 1
    # 字符串 "30" 被服务端强转成 30，客户端规范化比较仍判匹配
    assert ack.effective["retrieval"]["default_top_k"] == 30


async def test_reconfigure_bool_string_coercion():
    service = FakeService()
    async with _client(service) as client:
        ack = await client.reconfigure(flags={"llm_rerank_enabled": "true"})
    assert ack.effective["flags"]["llm_rerank_enabled"] is True


async def test_reconfigure_rerank_reports_reload():
    service = FakeService()
    async with _client(service) as client:
        ack = await client.reconfigure(rerank={"top_n": 5})
    assert ack.reranker_reloaded is True


async def test_reconfigure_422_rejected_with_code():
    service = FakeService()
    service.config_post_status = 422
    service.config_post_detail = {
        "message": "unknown retrieval field(s): nope",
        "code": "HOT_CONFIG_UNKNOWN_FIELD",
        "unknown": ["nope"],
    }
    async with _client(service) as client:
        with pytest.raises(ReconfigureRejected) as ei:
            await client.reconfigure(retrieval={"nope": 1})
    assert ei.value.code == "HOT_CONFIG_UNKNOWN_FIELD"
    assert ei.value.unknown == ["nope"]


async def test_reconfigure_409_gate():
    service = FakeService()
    service.config_post_status = 409
    service.config_post_detail = "hot retrieval config disabled"
    async with _client(service) as client:
        with pytest.raises(ReconfigureRejected) as ei:
            await client.reconfigure(retrieval={"default_top_k": 30})
    assert ei.value.status_code == 409


async def test_reconfigure_detects_generation_no_advance():
    """服务端 generation 没前进（静默 no-op）-> 客户端必须识破并抛错。"""
    service = FakeService()

    def frozen_generation(request):
        # POST 也返回 generation 0（模拟未推进）
        if request.url.path == "/admin/bench/retrieval-config" and request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "generation": 0,
                    "effective": service.effective,
                    "reranker_reloaded": None,
                },
            )
        return service.handler(request)

    client = BenchClient(
        "http://bench.test", "sk", transport=httpx.MockTransport(frozen_generation)
    )
    async with client:
        with pytest.raises(RuntimeError, match="did not advance generation"):
            await client.reconfigure(retrieval={"default_top_k": 30})


async def test_reconfigure_detects_effective_mismatch():
    """POST 声称改了、但 GET 回来的 effective 值不符 -> 识破并抛错。"""
    service = FakeService()

    def lying_server(request):
        if request.url.path == "/admin/bench/retrieval-config":
            if request.method == "POST":
                # POST 谎报 effective 已改成 99，但 GET 仍返回旧值 50
                service.config_generation += 1
                return httpx.Response(
                    200,
                    json={
                        "generation": service.config_generation,
                        "effective": {
                            "retrieval": {"default_top_k": 99},
                            "flags": {},
                            "milvus": {},
                            "rerank": {},
                        },
                        "reranker_reloaded": None,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "generation": service.config_generation,
                    "effective": {
                        "retrieval": {"default_top_k": 50},  # 旧值，未真改
                        "flags": {},
                        "milvus": {},
                        "rerank": {},
                    },
                    "reranker_reloaded": None,
                },
            )
        return service.handler(request)

    client = BenchClient(
        "http://bench.test", "sk", transport=httpx.MockTransport(lying_server)
    )
    async with client:
        with pytest.raises(RuntimeError, match="mismatch"):
            await client.reconfigure(retrieval={"default_top_k": 99})


async def test_reconfigure_verify_disabled_skips_reads():
    """verify=False 时不做前后 GET（裸下发，调试用）。"""
    service = FakeService()
    async with _client(service) as client:
        ack = await client.reconfigure(retrieval={"default_top_k": 30}, verify=False)
    assert ack.generation == 1  # 服务端仍推进，但客户端不校验 effective
