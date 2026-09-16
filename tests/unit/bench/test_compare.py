"""compare 对比测试：加载 / baseline 选择 / 差异参数自动标注 / 逐题 delta / 曲线 / 渲染。

核心断言：① differing_params 只返回跨 run 取值不同的键（自动标注取代文件名约定）② baseline
默认最早一份、显式 id 精确匹配、找不到报错 ③ per_query_delta 算出每题相对 baseline 的增减
④ param_curve 按字段值排序、字段缺失/不可排序报错 ⑤ render_compare 出含各段的 markdown
⑥ load_runs 按 run_id 过滤、缺 id 报错、空目录报错。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from oce.bench.compare import (
    CompareError,
    CurvePoint,
    differing_params,
    find_baseline,
    load_runs,
    param_curve,
    per_query_delta,
    render_compare,
    summarize,
)
from oce.bench.harness import EvaluationRow, EvaluationRun, IndexOutcome
from oce.bench.runrecord import ParamSnapshot, build_run_record, save_record


def _row(qid: str, *, top1: int, ndcg: float, category: str = "cat") -> EvaluationRow:
    return EvaluationRow(
        query_id=qid, category=category, difficulty=1, query=f"q {qid}",
        expected_files=["a.py"], top_paths=["a.py"], formatted="Path: a.py",
        client_elapsed_ms=10, server_elapsed_ms=5, top1_score=top1, ndcg_score=ndcg,
    )


def _run(rows: list[EvaluationRow]) -> EvaluationRun:
    return EvaluationRun(
        rows=rows,
        index=IndexOutcome(blob_names=["n"], uploaded=1, skipped=[], reused=False),
        peak_rss_mb=10.0, wall_seconds=1.0,
        client_latencies_ms=[r.client_elapsed_ms for r in rows],
    )


def _snapshot(top_k: int, generation: int, **over) -> ParamSnapshot:
    base = dict(
        embed_model="f2llm-v2-0.6b", embed_dimensions=1024,
        embed_endpoint="http://e", db_dialect="sqlite+aiosqlite", milvus_mode="lite",
        generation=generation,
        effective={
            "retrieval": {"default_top_k": top_k, "rrf_k": 60},
            "flags": {"rerank_enabled": False},
            "milvus": {"hnsw_ef_search": 512},
            "rerank": {},
        },
        pipeline="base",
    )
    base.update(over)
    return ParamSnapshot(**base)


def _record(top_k: int, gen: int, score_top1: int, ndcg: float) -> "object":
    return build_run_record(
        run=_run([_row("Q01", top1=score_top1, ndcg=ndcg), _row("Q02", top1=0, ndcg=0.0)]),
        params=_snapshot(top_k, gen),
        repo_name="flask", repo_root=Path("/flask"), repo_commit="c", repo_dirty=False,
        queries_path=Path("q.jsonl"), profile="local", tag="sweep",
        base_url="http://x",
        created_at=datetime(2026, 9, 15, 0, 0, gen, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# differing_params：自动标注
# ---------------------------------------------------------------------------


def test_differing_params_only_returns_varying_keys():
    records = [_record(30, 1, 1, 0.5), _record(80, 2, 0, 0.2)]
    differing = differing_params(records)
    # default_top_k 不同（30 vs 80）、generation 不同 -> 在结果里
    assert "retrieval.default_top_k" in differing
    assert differing["retrieval.default_top_k"] == {
        records[0].run_id: 30, records[1].run_id: 80,
    }
    # generation 是每次 reconfigure 都前进的单调计数器 -> 刻意排除，否则每次 sweep 都会
    # 把"generation 变了"当噪声列出来（它不表达"调了什么"）。
    assert "generation" not in differing
    # rrf_k 两组都 60（恒定）-> 不出现在 differing 里
    assert "retrieval.rrf_k" not in differing


def test_differing_params_single_record_empty():
    assert differing_params([_record(30, 1, 1, 0.5)]) == {}


def test_differing_params_detects_l2_model_change():
    a = _record(30, 1, 1, 0.5)
    b = build_run_record(
        run=_run([_row("Q01", top1=1, ndcg=0.5)]),
        params=_snapshot(30, 2, embed_model="other-model", embed_dimensions=768),
        repo_name="flask", repo_root=Path("/f"), repo_commit="c", repo_dirty=False,
        queries_path=Path("q.jsonl"), profile="local", tag="sweep", base_url="http://x",
        created_at=datetime(2026, 9, 15, 0, 0, 3, tzinfo=timezone.utc),
    )
    differing = differing_params([a, b])
    assert "embed_model" in differing
    assert "embed_dimensions" in differing


# ---------------------------------------------------------------------------
# baseline 选择
# ---------------------------------------------------------------------------


def test_find_baseline_defaults_to_earliest():
    records = [_record(30, 1, 1, 0.5), _record(80, 2, 0, 0.2)]
    assert find_baseline(records, None) is records[0]


def test_find_baseline_explicit_id():
    records = [_record(30, 1, 1, 0.5), _record(80, 2, 0, 0.2)]
    assert find_baseline(records, records[1].run_id) is records[1]


def test_find_baseline_unknown_id_raises():
    records = [_record(30, 1, 1, 0.5)]
    with pytest.raises(CompareError, match="not among loaded runs"):
        find_baseline(records, "no-such-run")


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def test_summarize_rows_have_scores_and_pipeline():
    records = [_record(30, 1, 1, 0.5), _record(80, 2, 0, 0.2)]
    rows = summarize(records)
    assert len(rows) == 2
    assert rows[0].pipeline == "base"
    assert rows[0].embed_model == "f2llm-v2-0.6b"
    assert rows[0].dimensions == 1024
    assert rows[0].generation == 1
    # top1_rate = 1/2 * 100
    assert rows[0].top1_rate == pytest.approx(50.0)
    # ndcg_mean = (0.5 + 0.0)/2
    assert rows[0].ndcg_mean == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# per_query_delta
# ---------------------------------------------------------------------------


def test_per_query_delta_vs_baseline():
    base = _record(30, 1, 1, 1.0)  # Q01: top1=1 + ndcg=1.0 -> total 2.0
    other = _record(80, 2, 0, 0.0)  # Q01: total 0.0
    deltas = per_query_delta([base, other], base)
    q01 = next(d for d in deltas if d.query_id == "Q01")
    assert q01.baseline_total == pytest.approx(2.0)
    assert q01.totals[other.run_id] == pytest.approx(0.0)
    assert q01.deltas[other.run_id] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# param_curve
# ---------------------------------------------------------------------------


def test_param_curve_sorts_by_value():
    # 乱序写入，曲线按 default_top_k 升序
    records = [_record(80, 2, 0, 0.2), _record(30, 1, 1, 0.5), _record(50, 3, 1, 0.3)]
    curve = param_curve(records, "retrieval.default_top_k")
    assert [p.value for p in curve] == [30, 50, 80]
    assert all(isinstance(p, CurvePoint) for p in curve)


def test_param_curve_missing_field_raises():
    records = [_record(30, 1, 1, 0.5)]
    with pytest.raises(CompareError, match="not present"):
        param_curve(records, "retrieval.nonexistent")


def test_param_curve_unsortable_raises():
    # 一个 record 的 top_k 是 None -> 混合类型不可排序
    a = _record(30, 1, 1, 0.5)
    b = build_run_record(
        run=_run([_row("Q01", top1=0, ndcg=0.0)]),
        params=_snapshot(30, 2),  # 先建一个正常的
        repo_name="flask", repo_root=Path("/f"), repo_commit="c", repo_dirty=False,
        queries_path=Path("q.jsonl"), profile="local", tag="sweep", base_url="http://x",
        created_at=datetime(2026, 9, 15, 0, 0, 3, tzinfo=timezone.utc),
    )
    # 手动把 b 的 top_k 改成 None 制造不可排序
    b.params.effective["retrieval"]["default_top_k"] = None
    with pytest.raises(CompareError, match="not sortable"):
        param_curve([a, b], "retrieval.default_top_k")


# ---------------------------------------------------------------------------
# render_compare
# ---------------------------------------------------------------------------


def test_render_compare_contains_all_sections():
    records = [_record(30, 1, 1, 0.5), _record(80, 2, 0, 0.2)]
    baseline = find_baseline(records, None)
    md = render_compare(records, baseline)
    assert "# OCE Retrieval Comparison" in md
    assert "## Varying parameters" in md
    assert "## Summary" in md
    assert "## Per-query delta" in md
    assert baseline.run_id in md
    assert "**(base)**" in md
    # 自动标注列出 default_top_k
    assert "retrieval.default_top_k" in md


def test_render_compare_with_curve():
    records = [_record(30, 1, 1, 0.5), _record(80, 2, 0, 0.2)]
    baseline = find_baseline(records, None)
    md = render_compare(records, baseline, param_field="retrieval.default_top_k")
    assert "## Curve:" in md
    assert "retrieval.default_top_k" in md


def test_render_compare_single_run_no_delta():
    records = [_record(30, 1, 1, 0.5)]
    md = render_compare(records, records[0])
    assert "only the baseline run is present" in md
    # 单 run 无差异参数
    assert "no L0/L1/L2 parameter differs" in md


# ---------------------------------------------------------------------------
# load_runs
# ---------------------------------------------------------------------------


def test_load_runs_from_dir(tmp_path: Path):
    for top_k, gen in ((30, 1), (80, 2)):
        save_record(_record(top_k, gen, 1, 0.5), tmp_path)
    records = load_runs(tmp_path)
    assert len(records) == 2
    # 按 run_id 排序
    assert [r.run_id for r in records] == sorted(r.run_id for r in records)


def test_load_runs_filter_by_run_id(tmp_path: Path):
    r1 = _record(30, 1, 1, 0.5)
    r2 = _record(80, 2, 0, 0.2)
    save_record(r1, tmp_path)
    save_record(r2, tmp_path)
    records = load_runs(tmp_path, run_ids=[r2.run_id])
    assert len(records) == 1
    assert records[0].run_id == r2.run_id


def test_load_runs_missing_id_raises(tmp_path: Path):
    save_record(_record(30, 1, 1, 0.5), tmp_path)
    with pytest.raises(CompareError, match="not found"):
        load_runs(tmp_path, run_ids=["ghost-run"])


def test_load_runs_empty_dir_raises(tmp_path: Path):
    with pytest.raises(CompareError, match="no run records"):
        load_runs(tmp_path)
