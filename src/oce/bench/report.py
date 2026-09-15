"""评测报告的 markdown 渲染（纯函数，不做 I/O 之外的副作用）。

从 oce-benchmark/scripts/run_retrieval_eval.py 的 render_report 移植，两处改进：

1. **解耦 git 调用**：原脚本把 ``subprocess git rev-parse`` 塞进 render_report，让纯渲染
   函数依赖子进程与文件系统。这里改为接受 ``repo_commit`` 参数，由调用方（cli/service）
   取好传入 —— 便于测试，且 Commit 7 的 RunRecord 会统一供给仓库元数据。
2. **补 p50/p95 时延**：harness 已逐题收集 client wall time，原报告却没呈现。热调参要
   "看效果"，时延与分数同等重要（改大 top_k 会涨分也会涨延迟），故显式列出。

本模块只渲染**单仓**报告。Commit 7 会把渲染源从 EvaluationRun 切换到 RunRecord（结构化
单一真源），使 md 与 json 永不漂移；届时此处的表格函数被 RunRecord 渲染路径复用。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from oce.bench.harness import EvaluationRow, EvaluationRun
from oce.bench.scoring import NDCG_K, POINTS_PER_QUERY


def percentile(values: Sequence[int | float], pct: float) -> float:
    """最近秩百分位（pct 取 0..100）。空序列返回 0.0。

    用最近秩而非线性插值：时延样本量小（100 题），插值会造出不存在的"分数毫秒"，
    最近秩更诚实地反映实际观测到的某次耗时。
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    # rank 落在 [0, n-1]；pct/100 * (n-1) 四舍五入取整
    rank = round((pct / 100.0) * (len(ordered) - 1))
    rank = max(0, min(len(ordered) - 1, rank))
    return float(ordered[rank])


@dataclass(frozen=True)
class ReportTotals:
    """一份报告的聚合分数（渲染 Total / Categories 表的数据源）。"""

    query_count: int
    top1_total: int
    ndcg_total: float

    @property
    def earned(self) -> float:
        return self.top1_total + self.ndcg_total

    @property
    def max_points(self) -> float:
        return self.query_count * POINTS_PER_QUERY

    @property
    def top1_rate(self) -> float:
        return (self.top1_total / self.query_count * 100) if self.query_count else 0.0

    @property
    def ndcg_mean(self) -> float:
        return (self.ndcg_total / self.query_count) if self.query_count else 0.0

    @property
    def score_pct(self) -> float:
        return (self.earned / self.max_points * 100) if self.max_points else 0.0


def compute_totals(rows: Sequence[EvaluationRow]) -> ReportTotals:
    return ReportTotals(
        query_count=len(rows),
        top1_total=sum(row.top1_score for row in rows),
        ndcg_total=sum(row.ndcg_score for row in rows),
    )


def compute_by_category(
    rows: Sequence[EvaluationRow],
) -> dict[str, tuple[int, int, float]]:
    """按分类聚合：``category -> (题数, top1 命中数, ndcg 累加)``。"""
    counts: dict[str, int] = {}
    top1: dict[str, int] = {}
    ndcg: dict[str, float] = {}
    for row in rows:
        counts[row.category] = counts.get(row.category, 0) + 1
        top1[row.category] = top1.get(row.category, 0) + row.top1_score
        ndcg[row.category] = ndcg.get(row.category, 0.0) + row.ndcg_score
    return {
        cat: (counts[cat], top1.get(cat, 0), ndcg.get(cat, 0.0))
        for cat in sorted(counts)
    }


def render_report(
    run: EvaluationRun,
    *,
    base_url: str,
    repo_root: Path,
    queries_path: Path | None = None,
    repo_commit: str | None = None,
    date_str: str = "",
    extra_header_lines: Sequence[str] = (),
) -> str:
    """把一次评测渲染成 markdown 文本（返回字符串，不落盘 —— 落盘由调用方决定）。

    Args:
        run: harness.evaluate 的产物。
        base_url / repo_root / queries_path: 报告头部的溯源信息。
        repo_commit: 被测仓库 HEAD（调用方取好传入；None 显示 "uncommitted"）。
        date_str: 报告日期（调用方用 ``datetime.now(timezone.utc)`` 传入，避免此纯函数
            依赖时钟；空则不显示 Date 行）。
        extra_header_lines: 额外头部行 —— Commit 7 用它注入 model / pipeline / profile /
            generation（补上原报告"不记录用的哪个模型/什么 pipeline"的缺口）。
    """
    rows = run.rows
    totals = compute_totals(rows)
    by_category = compute_by_category(rows)
    index = run.index

    latencies = run.client_latencies_ms
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)

    commit_display = repo_commit or "uncommitted"
    lines: list[str] = [
        "# OCE Retrieval Evaluation",
        "",
    ]
    if date_str:
        lines.append(f"- Date: {date_str}")
    lines.extend(
        [
            f"- Service: `{base_url}`",
            f"- Repository: `{repo_root}`",
            f"- Repository SHA: `{commit_display}`",
        ]
    )
    if queries_path is not None:
        lines.append(f"- Queries: `{queries_path.name}` ({totals.query_count})")
    else:
        lines.append(f"- Queries: {totals.query_count}")
    lines.extend(list(extra_header_lines))
    lines.extend(
        [
            f"- Index: {'reused' if index.reused else 'fresh upload'}; "
            f"source blobs considered: {index.uploaded}",
            f"- Upload failures: {len(index.skipped)}",
            f"- Peak process RSS: {run.peak_rss_mb:.1f} MB",
            f"- Wall time: {run.wall_seconds:.1f} s",
            f"- Query latency (client): p50 {p50:.0f} ms / p95 {p95:.0f} ms",
            f"- Scoring: Top-1 (1 pt, ranking ceiling) + nDCG@{NDCG_K} "
            f"(0..1 pt, result usability)",
            "",
            "## Total",
            "",
            f"| Points / {totals.max_points:.0f} | Top-1 | Mean nDCG@{NDCG_K} | Score % |",
            "|---:|---:|---:|---:|",
            f"| {totals.earned:.2f} | {totals.top1_total} / {totals.query_count} "
            f"({totals.top1_rate:.1f}%) | {totals.ndcg_mean:.3f} | "
            f"{totals.score_pct:.1f}% |",
            "",
            "## Categories",
            "",
            f"| Category | Top-1 | Mean nDCG@{NDCG_K} | Points |",
            "|---|---:|---:|---:|",
        ]
    )
    for category, (count, top1, ndcg_sum) in by_category.items():
        lines.append(
            f"| {category} | {top1} / {count} | {ndcg_sum / count:.3f} | "
            f"{top1 + ndcg_sum:.2f} / {count * POINTS_PER_QUERY} |"
        )

    error_rows = [row for row in rows if row.error]
    if error_rows:
        lines.extend(["", "## Query errors", ""])
        lines.extend(
            f"- `{row.query_id}`: {row.error}" for row in error_rows[:100]
        )

    if index.skipped:
        lines.extend(["", "## Skipped files", ""])
        lines.extend(f"- `{item}`" for item in index.skipped[:100])

    lines.extend(["", "## Details", ""])
    for row in rows:
        lines.extend(
            [
                f"### {row.query_id} ({row.category}, difficulty {row.difficulty})",
                f"- Query: {row.query}",
                f"- Expected: {', '.join(row.expected_files)}",
                f"- Top-{NDCG_K}: {', '.join(row.top_paths[:NDCG_K]) or '(no hits)'}",
                f"- Top-1: {row.top1_score} | nDCG@{NDCG_K}: {row.ndcg_score:.3f} | "
                f"Points: {row.total:.2f} / {POINTS_PER_QUERY}",
                f"- Elapsed: {row.client_elapsed_ms} ms (client) / "
                f"{row.server_elapsed_ms} ms (server)",
                "",
            ]
        )
        if row.error:
            lines.append(f"- **Error**: {row.error}")
            lines.append("")

    return "\n".join(lines)


def write_report(output: Path, markdown: str) -> None:
    """把渲染好的 markdown 落盘（建父目录）。落盘与渲染分离，便于测试只验渲染。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
