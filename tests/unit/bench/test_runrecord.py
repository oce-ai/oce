"""RunRecord 结构化记录测试：构造 / JSON round-trip / run_id 唯一性 / pipeline token。

核心断言：① to_dict→from_dict 无损还原（compare 依赖）② 同一次 sweep 内多组参数**不撞名**
（靠 generation 区分，时间戳秒级不够）③ pipeline token 按"最高启用层"派生（kNN-intent 已丢弃）
④ by_difficulty 键按数值排序 ⑤ save/load round-trip 落盘可读回。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from oce.bench.harness import EvaluationRow, EvaluationRun, IndexOutcome
from oce.bench.runrecord import (
    ParamSnapshot,
    RunRecord,
    build_run_record,
    compute_by_difficulty,
    list_records,
    load_record,
    make_run_id,
    pipeline_token,
    save_record,
)


def _row(
    qid: str,
    *,
    category: str = "cat",
    difficulty: int = 1,
    top1: int = 1,
    ndcg: float = 0.5,
    error: str | None = None,
) -> EvaluationRow:
    return EvaluationRow(
        query_id=qid,
        category=category,
        difficulty=difficulty,
        query=f"q {qid}",
        expected_files=["a.py"],
        top_paths=["a.py"],
        formatted="Path: a.py",
        client_elapsed_ms=12,
        server_elapsed_ms=8,
        top1_score=top1,
        ndcg_score=ndcg,
        error=error,
    )


def _run(rows: list[EvaluationRow], *, reused: bool = False) -> EvaluationRun:
    return EvaluationRun(
        rows=rows,
        index=IndexOutcome(blob_names=["n1"], uploaded=1, skipped=[], reused=reused),
        peak_rss_mb=123.4,
        wall_seconds=2.5,
        client_latencies_ms=[r.client_elapsed_ms for r in rows],
    )


def _snapshot(**over) -> ParamSnapshot:
    base = dict(
        embed_model="f2llm-v2-0.6b",
        embed_dimensions=1024,
        embed_endpoint="http://127.0.0.1:8994/v1/embeddings",
        db_dialect="sqlite+aiosqlite",
        milvus_mode="lite",
        generation=7,
        effective={
            "retrieval": {"default_top_k": 30},
            "flags": {"rerank_enabled": False, "llm_rerank_enabled": False},
            "milvus": {"hnsw_ef_search": 512},
            "rerank": {},
        },
        pipeline="base",
    )
    base.update(over)
    return ParamSnapshot(**base)


_FIXED = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# pipeline token
# ---------------------------------------------------------------------------


def test_pipeline_token_highest_layer_wins():
    # llm_rerank 最高 -> llmrerank（即使 rerank 也开）
    assert pipeline_token({"llm_rerank_enabled": True, "rerank_enabled": True}) == "llmrerank"
    # 仅 rerank
    assert pipeline_token({"rerank_enabled": True}) == "rerank"
    # 仅 intent 分类（kNN-intent 已丢弃，token 仍是 llmintent）
    assert pipeline_token({"intent_classification_enabled": True}) == "llmintent"
    # 全关
    assert pipeline_token({}) == "base"


def test_pipeline_token_precedence_over_intent():
    # intent + rerank 同时开 -> rerank 优先（按定义顺序 llmrerank > rerank > intent）
    flags = {"intent_classification_enabled": True, "rerank_enabled": True}
    assert pipeline_token(flags) == "rerank"


# ---------------------------------------------------------------------------
# run_id 唯一性（generation 区分同一次 sweep 的多组）
# ---------------------------------------------------------------------------


def test_run_id_includes_generation_for_uniqueness():
    a = make_run_id(
        created_at=_FIXED, repo_name="flask", pipeline="base",
        embed_model="m", dimensions=1024, tag="sweep", generation=1,
    )
    b = make_run_id(
        created_at=_FIXED, repo_name="flask", pipeline="base",
        embed_model="m", dimensions=1024, tag="sweep", generation=2,
    )
    # 同一秒、同一 repo/pipeline/model/dim/tag，仅 generation 不同 -> run_id 必须不同
    assert a != b
    assert "__g1__" in a and "__g2__" in b
    assert a.startswith("20260915T120000Z__flask__base__m__d1024")


def test_run_id_sanitizes_model_path_chars():
    rid = make_run_id(
        created_at=_FIXED, repo_name="r", pipeline="base",
        embed_model="org/model:v1", dimensions=512, tag="t", generation=0,
    )
    assert "/" not in rid and ":" not in rid
    assert "org-model-v1" in rid


# ---------------------------------------------------------------------------
# by_difficulty
# ---------------------------------------------------------------------------


def test_compute_by_difficulty_sorted_numeric_keys():
    rows = [
        _row("q3", difficulty=3), _row("q1a", difficulty=1),
        _row("q1b", difficulty=1, top1=0, ndcg=0.0), _row("q10", difficulty=10),
    ]
    out = compute_by_difficulty(rows)
    # 键按数值排序（1 < 3 < 10），而非字典序（"10" < "3"）
    assert list(out) == ["1", "3", "10"]
    # difficulty=1 有 2 题、1 个 top1 命中
    assert out["1"] == [2.0, 1.0, 0.5]


# ---------------------------------------------------------------------------
# build + round-trip
# ---------------------------------------------------------------------------


def test_build_run_record_aggregates_scores():
    rows = [_row("q1", top1=1, ndcg=1.0), _row("q2", top1=0, ndcg=0.5)]
    record = build_run_record(
        run=_run(rows),
        params=_snapshot(generation=3),
        repo_name="flask",
        repo_root=Path("/repos/flask"),
        repo_commit="abc123",
        repo_dirty=False,
        queries_path=Path("q.jsonl"),
        profile="local",
        tag="run",
        base_url="http://x",
        created_at=_FIXED,
    )
    assert record.query_count == 2
    assert record.top1_total == 1
    assert record.earned == pytest.approx(1.0 + 1.0 + 0.0 + 0.5)  # top1 + ndcg
    assert record.max_points == 4.0  # 2 题 * 2 分
    assert record.score_pct == pytest.approx(2.5 / 4.0 * 100)
    assert "__g3__" in record.run_id
    assert record.md_filename == f"{record.run_id}.md"
    assert record.repo_commit == "abc123"
    assert record.repo_dirty is False


def test_build_run_record_per_query_drops_formatted():
    rows = [_row("q1")]
    record = build_run_record(
        run=_run(rows), params=_snapshot(), repo_name="r",
        repo_root=Path("/r"), repo_commit=None, repo_dirty=True,
        queries_path=Path("q.jsonl"), profile="p", tag="t",
        base_url="http://x", created_at=_FIXED,
    )
    pq = record.per_query[0]
    assert "formatted" not in pq  # 全文不落盘（太大）
    assert pq["query_id"] == "q1"
    assert pq["total"] == pytest.approx(1.5)  # top1=1 + ndcg=0.5


def test_run_record_json_round_trip():
    rows = [_row("q1"), _row("q2", difficulty=2)]
    record = build_run_record(
        run=_run(rows), params=_snapshot(generation=5), repo_name="r",
        repo_root=Path("/r"), repo_commit="deadbeef", repo_dirty=False,
        queries_path=Path("q.jsonl"), profile="p", tag="t",
        base_url="http://x", created_at=_FIXED,
    )
    restored = RunRecord.from_dict(json.loads(json.dumps(record.to_dict())))
    assert restored.run_id == record.run_id
    assert restored.params.generation == 5
    assert restored.params.effective == record.params.effective
    assert isinstance(restored.params, ParamSnapshot)
    assert restored.score_pct == record.score_pct
    assert restored.by_difficulty == record.by_difficulty


def test_save_and_load_record_round_trip(tmp_path: Path):
    rows = [_row("q1")]
    record = build_run_record(
        run=_run(rows), params=_snapshot(), repo_name="r",
        repo_root=Path("/r"), repo_commit=None, repo_dirty=False,
        queries_path=Path("q.jsonl"), profile="p", tag="t",
        base_url="http://x", created_at=_FIXED,
    )
    path = save_record(record, tmp_path)
    assert path.name == f"{record.run_id}.json"
    assert path.exists()

    loaded = load_record(path)
    assert loaded.run_id == record.run_id
    assert loaded.params.embed_dimensions == 1024

    listed = list_records(tmp_path)
    assert len(listed) == 1
    assert listed[0].run_id == record.run_id


def test_list_records_sorted_by_run_id(tmp_path: Path):
    # 写三份不同 generation 的 record，list 按 run_id（含时间戳/generation）排序
    for gen in (3, 1, 2):
        record = build_run_record(
            run=_run([_row(f"q{gen}")]), params=_snapshot(generation=gen),
            repo_name="r", repo_root=Path("/r"), repo_commit=None,
            repo_dirty=False, queries_path=Path("q.jsonl"), profile="p",
            tag="t", base_url="http://x", created_at=_FIXED,
        )
        save_record(record, tmp_path)
    listed = list_records(tmp_path)
    gens = [r.params.generation for r in listed]
    assert gens == sorted(gens)  # generation 段在 run_id 里 -> 排序稳定


def test_param_snapshot_from_dict_ignores_unknown_keys():
    snap = ParamSnapshot.from_dict({"embed_model": "m", "bogus": 1})
    assert snap.embed_model == "m"
    assert not hasattr(snap, "bogus")
