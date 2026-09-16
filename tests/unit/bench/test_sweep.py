"""sweep 编排测试：参数路由 + 矩阵加载 + 全扫描状态机（MockTransport，不起真服务）。

覆盖三块：
1. **route_param / parse_param_args**：``--param KEY=VALUE`` 按 Commit 2 白名单路由到对应组；
   拼错/越层的 key 报错（绝不静默 no-op）。
2. **load_matrix**：声明式 ``[[set]]`` TOML 的结构与归属校验（未知顶层键 / 未知组 / 不可热改
   的键 / 键放错组 / 空矩阵），以及无名 set 的标签派生。
3. **run_sweep 状态机**：索引一次（最贵动作全流程仅一次）→ 每组参数热改（read-after-write
   校验 generation 前进 + effective⊇patch）→ 跑查询评分 → 落 RunRecord(JSON+MD) → 下一组。
   断言：上传只发生一次、generation 单调前进、每组各产一份带不同 run_id 的 JSON+MD、
   空 param set 不下发只读当前配置。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from oce.bench.client import BenchClient
from oce.bench.runrecord import load_record
from oce.bench.sweep import (
    ParamSet,
    SweepContext,
    SweepError,
    load_matrix,
    param_set_from_args,
    parse_param_args,
    route_param,
    run_sweep,
)


# ---------------------------------------------------------------------------
# route_param / parse_param_args
# ---------------------------------------------------------------------------


def test_route_param_to_groups():
    assert route_param("default_top_k") == "retrieval"
    assert route_param("rrf_k") == "retrieval"
    assert route_param("rerank_enabled") == "flags"
    assert route_param("llm_rerank_enabled") == "flags"
    assert route_param("hnsw_ef_search") == "milvus"
    assert route_param("top_n") == "rerank"


def test_route_param_rejects_unknown_key():
    with pytest.raises(SweepError, match="unknown or non-hot-swappable"):
        route_param("totally_made_up")


def test_route_param_rejects_l1_l2_keys():
    # chunk_size / HNSW M / dense_dim 属 L1/L2，要 reindex/reset，不可热改 -> 报错而非静默
    for key in ("chunk_size", "hnsw_m", "dense_dim", "embed_model"):
        with pytest.raises(SweepError):
            route_param(key)


def test_parse_param_args_groups_and_keeps_strings():
    grouped = parse_param_args(
        ["default_top_k=30", "rerank_enabled=false", "hnsw_ef_search=512"]
    )
    assert grouped == {
        "retrieval": {"default_top_k": "30"},
        "flags": {"rerank_enabled": "false"},
        "milvus": {"hnsw_ef_search": "512"},
    }


def test_parse_param_args_drops_empty_groups():
    grouped = parse_param_args(["default_top_k=30"])
    assert set(grouped) == {"retrieval"}


def test_parse_param_args_rejects_missing_equals():
    with pytest.raises(SweepError, match="expects KEY=VALUE"):
        parse_param_args(["default_top_k"])


def test_parse_param_args_rejects_empty_key():
    with pytest.raises(SweepError, match="empty key"):
        parse_param_args(["=30"])


def test_param_set_from_args_empty():
    ps = param_set_from_args([], name="run")
    assert ps.is_empty()
    assert ps.patch_groups() == {}


# ---------------------------------------------------------------------------
# load_matrix
# ---------------------------------------------------------------------------


def test_load_matrix_basic(tmp_path: Path):
    matrix = tmp_path / "m.toml"
    matrix.write_text(
        """
[[set]]
name = "topk-30"
[set.retrieval]
default_top_k = 30
[set.flags]
rerank_enabled = true

[[set]]
[set.retrieval]
default_top_k = 80
""",
        encoding="utf-8",
    )
    sets = load_matrix(matrix)
    assert len(sets) == 2
    assert sets[0].name == "topk-30"
    assert sets[0].retrieval == {"default_top_k": 30}
    assert sets[0].flags == {"rerank_enabled": True}
    # 第二个无名 -> 派生标签含内容
    assert "default_top_k=80" in sets[1].name


def test_load_matrix_rejects_unknown_top_level(tmp_path: Path):
    p = tmp_path / "m.toml"
    p.write_text("[[set]]\nname='a'\n[[nope]]\nx=1\n", encoding="utf-8")
    with pytest.raises(SweepError, match="unknown top-level"):
        load_matrix(p)


def test_load_matrix_rejects_unknown_group(tmp_path: Path):
    p = tmp_path / "m.toml"
    p.write_text("[[set]]\n[set.bogus]\nx=1\n", encoding="utf-8")
    with pytest.raises(SweepError, match="unknown group"):
        load_matrix(p)


def test_load_matrix_rejects_non_hot_key(tmp_path: Path):
    p = tmp_path / "m.toml"
    p.write_text("[[set]]\n[set.retrieval]\nchunk_size = 6000\n", encoding="utf-8")
    with pytest.raises(SweepError):
        load_matrix(p)


def test_load_matrix_rejects_key_in_wrong_group(tmp_path: Path):
    # default_top_k 属 retrieval，放 [set.flags] -> 归属错误
    p = tmp_path / "m.toml"
    p.write_text("[[set]]\n[set.flags]\ndefault_top_k = 30\n", encoding="utf-8")
    with pytest.raises(SweepError, match="belongs to"):
        load_matrix(p)


def test_load_matrix_empty_rejected(tmp_path: Path):
    p = tmp_path / "m.toml"
    p.write_text("# nothing here\n", encoding="utf-8")
    with pytest.raises(SweepError, match="at least one"):
        load_matrix(p)


def test_load_matrix_missing_file(tmp_path: Path):
    with pytest.raises(SweepError, match="not found"):
        load_matrix(tmp_path / "absent.toml")


# ---------------------------------------------------------------------------
# run_sweep 状态机（MockTransport 假服务）
# ---------------------------------------------------------------------------


class SweepFakeService:
    """假评测服务：可编程上传/嵌入/检索/热改，记录关键事件供断言。"""

    def __init__(self) -> None:
        self.upload_calls = 0  # 断言"索引只发生一次"
        self.retrieve_calls = 0
        self.generation = 0
        self.effective = {
            "retrieval": {"default_top_k": 50, "rrf_k": 60},
            "flags": {"rerank_enabled": False, "llm_rerank_enabled": False},
            "milvus": {"hnsw_ef_search": 512},
            "rerank": {"top_n": 10, "min_score": 0.05},
        }
        # 检索结果按查询回显对应文件（"where is a" -> a.py），让每题都命中
        # -> top1=1, ndcg>0，便于断言聚合分数。
        self.retrieve_elapsed_ms = 7

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        body = json.loads(request.content) if request.content else {}

        if path == "/batch-upload":
            self.upload_calls += 1
            names = [f"name-{b['path']}" for b in body.get("blobs", [])]
            return httpx.Response(200, json={"blob_names": names})

        if path == "/find-missing":
            return httpx.Response(
                200, json={"unknown_memory_names": [], "nonindexed_blob_names": []}
            )

        if path == "/agents/blob-status":
            return httpx.Response(
                200,
                json={
                    "unknown_blob_names": [],
                    "nonindexed_blob_names": [],
                    "checkpoint_not_found": False,
                },
            )

        if path == "/agents/codebase-retrieval":
            self.retrieve_calls += 1
            query = body.get("information_request", "")
            # 查询以 "where is X" 结尾；回显 X.py 让每题命中
            hit = f"{query.rsplit(' ', 1)[-1]}.py"
            return httpx.Response(
                200,
                json={
                    "formatted_retrieval": f"Path: {hit}\nsnippet here",
                    "codebase_retrieval_elapsed_ms": self.retrieve_elapsed_ms,
                },
            )

        if path == "/admin/bench/retrieval-config":
            if request.method == "GET":
                return httpx.Response(
                    200,
                    json={
                        "generation": self.generation,
                        "effective": self.effective,
                        "reranker_reloaded": None,
                    },
                )
            self.generation += 1
            for group in ("retrieval", "flags", "milvus", "rerank"):
                for key, value in (body.get(group) or {}).items():
                    self.effective[group][key] = _coerce(value)
            return httpx.Response(
                200,
                json={
                    "generation": self.generation,
                    "effective": self.effective,
                    "reranker_reloaded": bool(body.get("rerank")),
                },
            )

        return httpx.Response(404, text=f"no route {path}")


def _coerce(value):
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


@pytest.fixture
def tiny_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (repo / "b.py").write_text("def b():\n    return 2\n", encoding="utf-8")
    return repo


@pytest.fixture
def queries_file(tmp_path: Path) -> Path:
    q = tmp_path / "queries.jsonl"
    lines = [
        {"id": "Q01", "category": "file_exact_match", "difficulty": 1,
         "query": "where is a", "expected_files": ["a.py"]},
        {"id": "Q02", "category": "file_exact_match", "difficulty": 2,
         "query": "where is b", "expected_files": ["b.py"]},
    ]
    q.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return q


def _context(tiny_repo: Path, queries_file: Path) -> SweepContext:
    return SweepContext(
        repo_name="repo",
        repo_root=tiny_repo,
        repo_commit="abc123",
        repo_dirty=False,
        queries_path=queries_file,
        profile_name="local",
        tag="sweep",
        embed_model="f2llm-v2-0.6b",
        embed_dimensions=1024,
        embed_endpoint="http://embed",
        db_dialect="sqlite+aiosqlite",
        milvus_mode="lite",
    )


def _client(service: SweepFakeService) -> BenchClient:
    return BenchClient(
        "http://bench.test", "sk", transport=httpx.MockTransport(service.handler),
        poll_interval=0.0,
    )


async def test_run_sweep_indexes_once_scores_each_set(tiny_repo, queries_file, tmp_path):
    """核心：上传只一次，两组参数各热改 + 评分 + 落一份 RunRecord。"""
    service = SweepFakeService()
    runs_dir = tmp_path / "runs"
    context = _context(tiny_repo, queries_file)
    param_sets = [
        ParamSet(name="topk-30", retrieval={"default_top_k": 30}),
        ParamSet(name="topk-80", retrieval={"default_top_k": 80}),
    ]
    async with _client(service) as client:
        records = await run_sweep(
            client, context, param_sets=param_sets, runs_dir=runs_dir,
            embedding_timeout=5.0,
        )

    # 索引（上传）全流程只发生一次，但跑了 2 组 * 2 题 = 4 次检索
    assert service.upload_calls == 1
    assert service.retrieve_calls == 4
    # 两组各产一份记录
    assert len(records) == 2
    assert records[0].run_id != records[1].run_id  # generation 区分，不撞名
    # 每组记录了各自热改后的 effective
    assert records[0].params.effective["retrieval"]["default_top_k"] == 30
    assert records[1].params.effective["retrieval"]["default_top_k"] == 80
    assert records[0].params.generation == 1
    assert records[1].params.generation == 2
    # 命中：每题 top1=1
    assert records[0].top1_total == 2
    assert records[0].query_count == 2


async def test_run_sweep_writes_json_and_md(tiny_repo, queries_file, tmp_path):
    """每组落 .json + .md 两个文件，md 头部含 model/pipeline/profile/generation。"""
    service = SweepFakeService()
    runs_dir = tmp_path / "runs"
    context = _context(tiny_repo, queries_file)
    async with _client(service) as client:
        records = await run_sweep(
            client, context,
            param_sets=[ParamSet(name="base", retrieval={"default_top_k": 40})],
            runs_dir=runs_dir, embedding_timeout=5.0,
        )
    record = records[0]
    json_path = runs_dir / f"{record.run_id}.json"
    md_path = runs_dir / record.md_filename
    assert json_path.exists()
    assert md_path.exists()
    # JSON 可读回（compare 的入口）
    assert load_record(json_path).run_id == record.run_id
    # md 头部补上了旧报告缺失的信息
    md = md_path.read_text(encoding="utf-8")
    assert "Model:" in md and "Pipeline:" in md
    assert "Profile:" in md and "Config generation:" in md
    assert record.run_id in md


async def test_run_sweep_empty_set_uses_current_config(tiny_repo, queries_file, tmp_path):
    """空 param set 不下发热改（generation 不前进），只读当前配置评分。"""
    service = SweepFakeService()
    runs_dir = tmp_path / "runs"
    context = _context(tiny_repo, queries_file)
    async with _client(service) as client:
        records = await run_sweep(
            client, context, param_sets=[ParamSet(name="current")],
            runs_dir=runs_dir, embedding_timeout=5.0,
        )
    # 没有 POST 热改 -> generation 停在 0
    assert service.generation == 0
    assert records[0].params.generation == 0
    # 仍跑了查询、落了记录
    assert records[0].query_count == 2


async def test_run_sweep_reuse_index_skips_upload(tiny_repo, queries_file, tmp_path):
    """reuse_index=True 走 /find-missing 而非 /batch-upload（上传计数为 0）。"""
    service = SweepFakeService()
    runs_dir = tmp_path / "runs"
    context = _context(tiny_repo, queries_file)
    async with _client(service) as client:
        await run_sweep(
            client, context, param_sets=[ParamSet(name="x", retrieval={"rrf_k": 70})],
            runs_dir=runs_dir, reuse_index=True, embedding_timeout=5.0,
        )
    assert service.upload_calls == 0


async def test_run_sweep_requires_param_sets(tiny_repo, queries_file, tmp_path):
    service = SweepFakeService()
    context = _context(tiny_repo, queries_file)
    async with _client(service) as client:
        with pytest.raises(SweepError, match="at least one param set"):
            await run_sweep(
                client, context, param_sets=[], runs_dir=tmp_path / "runs",
            )


async def test_run_sweep_detects_read_after_write_failure(tiny_repo, queries_file, tmp_path):
    """服务端谎报 generation 不前进 -> client 的 read-after-write 断言让 sweep 失败。"""
    service = SweepFakeService()

    def frozen_generation(request: httpx.Request) -> httpx.Response:
        if (
            request.url.path == "/admin/bench/retrieval-config"
            and request.method == "POST"
        ):
            return httpx.Response(
                200,
                json={
                    "generation": 0,  # 谎称没推进
                    "effective": service.effective,
                    "reranker_reloaded": None,
                },
            )
        return service.handler(request)

    client = BenchClient(
        "http://bench.test", "sk", transport=httpx.MockTransport(frozen_generation),
        poll_interval=0.0,
    )
    context = _context(tiny_repo, queries_file)
    async with client:
        with pytest.raises(RuntimeError, match="did not advance generation"):
            await run_sweep(
                client, context,
                param_sets=[ParamSet(name="x", retrieval={"default_top_k": 30})],
                runs_dir=tmp_path / "runs", embedding_timeout=5.0,
            )


def test_param_set_patch_groups_only_non_empty():
    ps = ParamSet(name="x", retrieval={"rrf_k": 60}, flags={})
    assert ps.patch_groups() == {"retrieval": {"rrf_k": 60}}
    assert not ParamSet(name="e").patch_groups()
