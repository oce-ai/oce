"""对比多份 run 记录：汇总矩阵 + 逐题 delta + 参数曲线。

旧机制：报告头部不记录用哪个模型/pipeline，只能靠文件名约定
（``{model}__{pipeline}__[repo]__d{dim}__{date}.md``），再用 ``compare_intent_*.py`` **正则从
markdown 反向刮分数**——脆弱、语义靠猜（附录一堆"缩写待确认"）。

新机制：RunRecord 的 JSON 里有完整 ParamSnapshot，compare 直接读结构化数据：
- ``summarize``：一行一份 run，含模型/维度/pipeline/generation + 分数 + 时延。
- ``differing_params``：跨 run **自动找出哪些 L0 参数值不同** —— 矩阵据此自动标注"每列改了
  什么"，不再依赖文件名。
- ``per_query_delta``：逐题分数 vs baseline，定位"哪组参数在哪类题上退化"。
- ``param_curve``：按某字段值排序出"参数→分数"曲线（如 top_k 30/50/80 的涨分与涨延迟）。

本模块全是纯函数（读 RunRecord、出结构/markdown），不做 I/O 之外的副作用，极易测。
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from oce.bench.runrecord import RunRecord, list_records


class CompareError(Exception):
    """对比失败：找不到 baseline、空记录集、字段不可排序、promote 目标缺失等。"""


# 受追踪的 golden baseline 子目录（相对仓库根）。sweep 产物（bench/runs/*）被 .gitignore，
# 但 curated 的长期 baseline 落在此处、随仓库走 —— 满足"run 机制进 git"又不让产物撑爆仓库。
GOLDEN_SUBDIR = Path("bench") / "runs" / "golden"


def promote_run(
    runs_dir: str | Path, run_id: str, golden_dir: str | Path | None = None
) -> list[Path]:
    """把一份 run 的 ``.json``（+ 配对的 ``.md`` 若有）复制进受追踪的 golden 目录。

    长期保留的 baseline 用它"晋升"出 .gitignore 的 sweep 产物区。只复制、不移动 —— 原
    runs_dir 的记录不动，便于继续参与本地 compare。返回复制出的目标路径列表。

    Args:
        runs_dir: 源 run 记录目录（``<run_id>.json`` 所在）。
        run_id: 要晋升的 run（精确匹配，不做前缀模糊）。
        golden_dir: 目标目录；None 用默认 ``bench/runs/golden``（相对 cwd）。
    """
    runs_dir = Path(runs_dir).expanduser()
    golden_dir = Path(golden_dir).expanduser() if golden_dir else GOLDEN_SUBDIR
    json_src = runs_dir / f"{run_id}.json"
    if not json_src.is_file():
        raise CompareError(
            f"run '{run_id}' not found in {runs_dir} (expected {json_src.name})"
        )
    golden_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    json_dst = golden_dir / json_src.name
    shutil.copy2(json_src, json_dst)
    copied.append(json_dst)
    md_src = runs_dir / f"{run_id}.md"
    if md_src.is_file():
        md_dst = golden_dir / md_src.name
        shutil.copy2(md_src, md_dst)
        copied.append(md_dst)
    return copied


# ---------------------------------------------------------------------------
# 加载 + baseline 选择
# ---------------------------------------------------------------------------


def load_runs(runs_dir: str | Path, *, run_ids: Sequence[str] | None = None) -> list[RunRecord]:
    """加载目录下（或指定 run_id 列表的）run 记录，按 run_id（含时间戳）排序。"""
    records = list_records(runs_dir)
    if run_ids:
        wanted = set(run_ids)
        records = [r for r in records if r.run_id in wanted]
        missing = wanted - {r.run_id for r in records}
        if missing:
            raise CompareError(f"run_id(s) not found in {runs_dir}: {sorted(missing)}")
    if not records:
        raise CompareError(f"no run records found in {runs_dir}")
    return records


def find_baseline(records: Sequence[RunRecord], baseline_id: str | None) -> RunRecord:
    """选 baseline：显式 id 则精确匹配；否则取最早一份（时间戳前缀 -> 首个）。"""
    if baseline_id is None:
        return records[0]
    for record in records:
        if record.run_id == baseline_id:
            return record
    raise CompareError(
        f"baseline '{baseline_id}' not among loaded runs; "
        f"available: {[r.run_id for r in records]}"
    )


# ---------------------------------------------------------------------------
# 自动标注：跨 run 找出哪些参数不同
# ---------------------------------------------------------------------------


def _flatten_params(record: RunRecord) -> dict[str, Any]:
    """把一份 record 的可比参数摊平成 ``{点分路径: 值}``。

    覆盖 L2（模型/维度/后端）、L1（HNSW）、以及 L0 effective 全集（retrieval/flags/milvus/
    rerank 四组的每个键）。differing_params 据此逐键比对，自动发现"这几份 run 到底改了什么"。
    """
    flat: dict[str, Any] = {
        "embed_model": record.params.embed_model,
        "embed_dimensions": record.params.embed_dimensions,
        "db_dialect": record.params.db_dialect,
        "milvus_mode": record.params.milvus_mode,
        "hnsw_m": record.params.hnsw_m,
        "hnsw_ef_construction": record.params.hnsw_ef_construction,
        "pipeline": record.params.pipeline,
        "profile": record.profile,
    }
    for group, values in record.params.effective.items():
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{group}.{key}"] = value
    return flat


def differing_params(records: Sequence[RunRecord]) -> dict[str, dict[str, Any]]:
    """跨 run 找出**值不恒定**的参数键：``{param_path: {run_id: value}}``。

    只返回在不同 run 间取值有差异的键（恒定键无对比价值，省略以保持矩阵紧凑）。这是取代
    文件名约定的核心：矩阵自动知道"这几列到底在比什么"。
    """
    if len(records) < 2:
        return {}
    flats = [(r.run_id, _flatten_params(r)) for r in records]
    all_keys = set().union(*(set(f) for _, f in flats))
    differing: dict[str, dict[str, Any]] = {}
    for key in sorted(all_keys):
        values = {run_id: flat.get(key) for run_id, flat in flats}
        if len(set(_hashable(v) for v in values.values())) > 1:
            differing[key] = values
    return differing


def _hashable(value: Any) -> Any:
    """把值转成可放进 set 去重的形态（dict/list 转 JSON 串）。"""
    if isinstance(value, (dict, list)):
        import json

        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


# ---------------------------------------------------------------------------
# 汇总矩阵
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SummaryRow:
    """一份 run 的汇总行。"""

    run_id: str
    pipeline: str
    embed_model: str
    dimensions: int
    generation: int
    score_pct: float
    top1_rate: float
    ndcg_mean: float
    latency_p50_ms: float
    latency_p95_ms: float
    index_reused: bool


def _summary_row(record: RunRecord) -> SummaryRow:
    top1_rate = (
        record.top1_total / record.query_count * 100 if record.query_count else 0.0
    )
    ndcg_mean = record.ndcg_total / record.query_count if record.query_count else 0.0
    return SummaryRow(
        run_id=record.run_id,
        pipeline=record.params.pipeline,
        embed_model=record.params.embed_model,
        dimensions=record.params.embed_dimensions,
        generation=record.params.generation,
        score_pct=record.score_pct,
        top1_rate=top1_rate,
        ndcg_mean=ndcg_mean,
        latency_p50_ms=record.latency_p50_ms,
        latency_p95_ms=record.latency_p95_ms,
        index_reused=record.index_reused,
    )


def summarize(records: Sequence[RunRecord]) -> list[SummaryRow]:
    return [_summary_row(r) for r in records]


# ---------------------------------------------------------------------------
# 逐题 delta vs baseline
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QueryDelta:
    """某题在某 run 相对 baseline 的分数变化。"""

    query_id: str
    category: str
    baseline_total: float
    totals: dict[str, float]  # run_id -> 该题得分
    deltas: dict[str, float]  # run_id -> (得分 - baseline)


def _index_per_query(record: RunRecord) -> dict[str, float]:
    return {q["query_id"]: float(q["total"]) for q in record.per_query}


def per_query_delta(
    records: Sequence[RunRecord], baseline: RunRecord
) -> list[QueryDelta]:
    """逐题：baseline 得分 vs 每份 run 得分 + delta。按 query_id 顺序（baseline 的题序）。"""
    totals_by_run = {r.run_id: _index_per_query(r) for r in records}
    baseline_totals = totals_by_run[baseline.run_id]
    # 题序与分类取自 baseline 的 per_query
    order = [(q["query_id"], q["category"]) for q in baseline.per_query]
    out: list[QueryDelta] = []
    for query_id, category in order:
        base = baseline_totals.get(query_id, 0.0)
        totals = {
            r.run_id: totals_by_run[r.run_id].get(query_id, 0.0) for r in records
        }
        deltas = {run_id: total - base for run_id, total in totals.items()}
        out.append(
            QueryDelta(
                query_id=query_id,
                category=category,
                baseline_total=base,
                totals=totals,
                deltas=deltas,
            )
        )
    return out


# ---------------------------------------------------------------------------
# 参数曲线
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurvePoint:
    """参数曲线上的一点：某参数取值 -> 分数。"""

    run_id: str
    value: Any
    score_pct: float
    top1_rate: float
    ndcg_mean: float
    latency_p50_ms: float


def param_curve(records: Sequence[RunRecord], param_field: str) -> list[CurvePoint]:
    """按 ``param_field``（点分路径，如 ``retrieval.default_top_k``）取值排序出曲线。

    取值必须可排序（数值/字符串）；含 None 或混合类型时报错，提示该字段不适合画曲线。
    """
    points: list[tuple[Any, CurvePoint]] = []
    for record in records:
        flat = _flatten_params(record)
        if param_field not in flat:
            raise CompareError(
                f"param '{param_field}' not present in run {record.run_id}; "
                f"known: {sorted(flat)}"
            )
        value = flat[param_field]
        summary = _summary_row(record)
        points.append(
            (
                value,
                CurvePoint(
                    run_id=record.run_id,
                    value=value,
                    score_pct=summary.score_pct,
                    top1_rate=summary.top1_rate,
                    ndcg_mean=summary.ndcg_mean,
                    latency_p50_ms=summary.latency_p50_ms,
                ),
            )
        )
    try:
        points.sort(key=lambda pv: pv[0])
    except TypeError as exc:
        raise CompareError(
            f"param '{param_field}' values are not sortable (mixed types / None): "
            f"{[v for v, _ in points]}"
        ) from exc
    return [point for _, point in points]


# ---------------------------------------------------------------------------
# markdown 渲染
# ---------------------------------------------------------------------------


def render_compare(
    records: Sequence[RunRecord],
    baseline: RunRecord,
    *,
    param_field: str | None = None,
) -> str:
    """渲染对比报告 markdown：汇总矩阵 + 差异参数标注 + 逐题 delta +（可选）参数曲线。"""
    lines: list[str] = ["# OCE Retrieval Comparison", ""]
    lines.append(f"- Baseline: `{baseline.run_id}`")
    lines.append(f"- Runs compared: {len(records)}")
    lines.append("")

    # --- 自动标注：这几份 run 到底在比什么 ---
    differing = differing_params(records)
    lines.append("## Varying parameters")
    lines.append("")
    if differing:
        lines.append("| Parameter | " + " | ".join(r.run_id for r in records) + " |")
        lines.append("|---" * (len(records) + 1) + "|")
        for key, values in differing.items():
            cells = " | ".join(_fmt(values.get(r.run_id)) for r in records)
            lines.append(f"| `{key}` | {cells} |")
    else:
        lines.append("_no L0/L1/L2 parameter differs across these runs_")
    lines.append("")

    # --- 汇总矩阵 ---
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Run | Pipeline | Model | Dim | Gen | Score % | Top-1 % | nDCG | p50 | p95 | Index |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in summarize(records):
        marker = " **(base)**" if row.run_id == baseline.run_id else ""
        lines.append(
            f"| `{row.run_id}`{marker} | {row.pipeline} | {row.embed_model} | "
            f"{row.dimensions} | {row.generation} | {row.score_pct:.1f} | "
            f"{row.top1_rate:.1f} | {row.ndcg_mean:.3f} | "
            f"{row.latency_p50_ms:.0f} | {row.latency_p95_ms:.0f} | "
            f"{'reused' if row.index_reused else 'fresh'} |"
        )
    lines.append("")

    # --- 参数曲线（可选）---
    if param_field:
        lines.append(f"## Curve: `{param_field}`")
        lines.append("")
        lines.append("| Value | Run | Score % | Top-1 % | nDCG | p50 ms |")
        lines.append("|---|---|---:|---:|---:|---:|")
        for point in param_curve(records, param_field):
            lines.append(
                f"| {_fmt(point.value)} | `{point.run_id}` | {point.score_pct:.1f} | "
                f"{point.top1_rate:.1f} | {point.ndcg_mean:.3f} | "
                f"{point.latency_p50_ms:.0f} |"
            )
        lines.append("")

    # --- 逐题 delta vs baseline ---
    lines.append(f"## Per-query delta vs `{baseline.run_id}`")
    lines.append("")
    others = [r for r in records if r.run_id != baseline.run_id]
    if others:
        header = "| Query | Cat | Base | " + " | ".join(
            f"Δ {r.run_id}" for r in others
        ) + " |"
        lines.append(header)
        lines.append("|---" * (3 + len(others)) + "|")
        for delta in per_query_delta(records, baseline):
            cells = " | ".join(_fmt_delta(delta.deltas[r.run_id]) for r in others)
            lines.append(
                f"| `{delta.query_id}` | {delta.category} | "
                f"{delta.baseline_total:.2f} | {cells} |"
            )
    else:
        lines.append("_only the baseline run is present; no delta to show_")
    lines.append("")

    return "\n".join(lines)


def _fmt(value: Any) -> str:
    """表格单元格式化：None -> 空，bool -> true/false，float 保留合理精度。"""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _fmt_delta(delta: float) -> str:
    """delta 带符号，零值显示 `±0`（无变化）便于扫读。"""
    if abs(delta) < 1e-9:
        return "±0.00"
    return f"{delta:+.2f}"
