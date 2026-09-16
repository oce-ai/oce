"""评测报告的 markdown 渲染（纯函数，不做 I/O 之外的副作用）。

从 oce-benchmark/scripts/run_retrieval_eval.py 的 render_report 移植，两处改进：

1. **解耦 git 调用**：原脚本把 ``subprocess git rev-parse`` 塞进 render_report，让纯渲染
   函数依赖子进程与文件系统。这里改为接受 ``repo_commit`` 参数，由调用方（cli/service）
   取好传入 —— 便于测试，且 RunRecord 会统一供给仓库元数据。
2. **补 p50/p95 时延**：harness 已逐题收集 client wall time，原报告却没呈现。热调参要
   "看效果"，时延与分数同等重要（改大 top_k 会涨分也会涨延迟），故显式列出。

**单一渲染体（Commit 7）**：``_render(view)`` 是唯一产出 markdown 的函数，由 ``_ReportView``
这一归一化中间结构喂入。两个公开入口都只是它的适配器：
- ``render_record(record)`` —— 从结构化 RunRecord 渲染。这是 sweep 的落盘路径，也是**离线
  重渲**路径（``oce bench report`` 读 .json 重新出 .md，无需活服务）。头部自动注入
  model/pipeline/profile/generation（补上旧报告"不记录用的哪个模型"的缺口）。
- ``render_report(run, ...)`` —— 从 EvaluationRun 渲染（ad-hoc / 测试用），缺的参数快照
  经 extra_header_lines 补。

两者共用 ``_render``，故 md 与 json 结构**永不可能漂移**——彻底告别"先写 md 再正则刮回分数"。
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


@dataclass(frozen=True)
class _DetailRow:
    """单题详情的归一化形态（EvaluationRow 与 RunRecord.per_query dict 都映射到它）。"""

    query_id: str
    category: str
    difficulty: int
    query: str
    expected_files: list[str]
    top_paths: list[str]
    top1_score: int
    ndcg_score: float
    total: float
    client_elapsed_ms: int
    server_elapsed_ms: int
    error: str | None

    @classmethod
    def from_row(cls, row: EvaluationRow) -> "_DetailRow":
        return cls(
            query_id=row.query_id, category=row.category, difficulty=row.difficulty,
            query=row.query, expected_files=list(row.expected_files),
            top_paths=list(row.top_paths), top1_score=row.top1_score,
            ndcg_score=row.ndcg_score, total=row.total,
            client_elapsed_ms=row.client_elapsed_ms,
            server_elapsed_ms=row.server_elapsed_ms, error=row.error,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "_DetailRow":
        return cls(
            query_id=data.get("query_id", "?"), category=data.get("category", "unknown"),
            difficulty=int(data.get("difficulty", 1)), query=data.get("query", ""),
            expected_files=list(data.get("expected_files", [])),
            top_paths=list(data.get("top_paths", [])),
            top1_score=int(data.get("top1_score", 0)),
            ndcg_score=float(data.get("ndcg_score", 0.0)),
            total=float(data.get("total", 0.0)),
            client_elapsed_ms=int(data.get("client_elapsed_ms", 0)),
            server_elapsed_ms=int(data.get("server_elapsed_ms", 0)),
            error=data.get("error"),
        )


@dataclass(frozen=True)
class _ReportView:
    """渲染 markdown 所需的**全部**归一化数据（两个入口都先转成它，再喂给 _render）。

    刻意做成纯数据：_render 只读它、不碰 EvaluationRun / RunRecord 的具体类型，于是
    "从内存对象渲染"与"从磁盘 JSON 离线重渲"走的是同一条代码路径，输出必然一致。
    """

    totals: ReportTotals
    by_category: dict[str, tuple[int, int, float]]
    details: list[_DetailRow]
    index_reused: bool
    uploaded: int
    skipped_reasons: list[str]
    peak_rss_mb: float
    wall_seconds: float
    latency_p50_ms: float
    latency_p95_ms: float
    # 头部溯源
    base_url: str
    repo_root: str
    repo_commit: str | None
    queries_path: str | None
    date_str: str = ""
    extra_header_lines: Sequence[str] = ()
    repo_dirty: bool = False


def _render(view: _ReportView) -> str:
    """唯一的 markdown 产出函数（纯渲染，读归一化 _ReportView）。返回字符串不落盘。"""
    totals = view.totals
    commit_display = view.repo_commit or "uncommitted"
    lines: list[str] = ["# OCE Retrieval Evaluation", ""]
    if view.date_str:
        lines.append(f"- Date: {view.date_str}")
    lines.extend(
        [
            f"- Service: `{view.base_url}`",
            f"- Repository: `{view.repo_root}`",
            f"- Repository SHA: `{commit_display}`",
        ]
    )
    if view.repo_dirty:
        lines.append("- Repository state: **dirty (uncommitted changes; score may not reproduce)**")
    queries_label = Path(view.queries_path).name if view.queries_path else None
    if queries_label:
        lines.append(f"- Queries: `{queries_label}` ({totals.query_count})")
    else:
        lines.append(f"- Queries: {totals.query_count}")
    lines.extend(list(view.extra_header_lines))
    lines.extend(
        [
            f"- Index: {'reused' if view.index_reused else 'fresh upload'}; "
            f"source blobs considered: {view.uploaded}",
            f"- Upload failures: {len(view.skipped_reasons)}",
            f"- Peak process RSS: {view.peak_rss_mb:.1f} MB",
            f"- Wall time: {view.wall_seconds:.1f} s",
            f"- Query latency (client): p50 {view.latency_p50_ms:.0f} ms / "
            f"p95 {view.latency_p95_ms:.0f} ms",
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
    for category, (count, top1, ndcg_sum) in view.by_category.items():
        lines.append(
            f"| {category} | {top1} / {count} | {ndcg_sum / count:.3f} | "
            f"{top1 + ndcg_sum:.2f} / {count * POINTS_PER_QUERY} |"
        )

    error_rows = [d for d in view.details if d.error]
    if error_rows:
        lines.extend(["", "## Query errors", ""])
        lines.extend(f"- `{d.query_id}`: {d.error}" for d in error_rows[:100])

    if view.skipped_reasons:
        lines.extend(["", "## Skipped files", ""])
        lines.extend(f"- `{item}`" for item in view.skipped_reasons[:100])

    lines.extend(["", "## Details", ""])
    for d in view.details:
        lines.extend(
            [
                f"### {d.query_id} ({d.category}, difficulty {d.difficulty})",
                f"- Query: {d.query}",
                f"- Expected: {', '.join(d.expected_files)}",
                f"- Top-{NDCG_K}: {', '.join(d.top_paths[:NDCG_K]) or '(no hits)'}",
                f"- Top-1: {d.top1_score} | nDCG@{NDCG_K}: {d.ndcg_score:.3f} | "
                f"Points: {d.total:.2f} / {POINTS_PER_QUERY}",
                f"- Elapsed: {d.client_elapsed_ms} ms (client) / "
                f"{d.server_elapsed_ms} ms (server)",
                "",
            ]
        )
        if d.error:
            lines.append(f"- **Error**: {d.error}")
            lines.append("")

    return "\n".join(lines)


def _by_category_from_record(by_category: dict[str, list[float]]) -> dict[str, tuple[int, int, float]]:
    """RunRecord.by_category（JSON 友好的 [count,top1,ndcg] 列表）-> report 的元组形态。"""
    return {
        cat: (int(vals[0]), int(vals[1]), float(vals[2]))
        for cat, vals in by_category.items()
    }


def render_record(record) -> str:
    """从结构化 RunRecord 渲染 markdown（离线重渲路径；无需活服务、无需 EvaluationRun）。

    头部自动注入 model / pipeline / profile / generation —— 补上旧报告"不记录用的哪个
    模型、什么 pipeline"的缺口（旧机制只能靠文件名约定 + 正则刮 markdown 反推）。

    record 用鸭子类型（不 import RunRecord，避免 report<->runrecord 循环 import）：需要
    params / per_query / by_category / 资源字段 / 溯源字段，均由 build_run_record 供给。
    """
    params = record.params
    totals = ReportTotals(
        query_count=record.query_count,
        top1_total=record.top1_total,
        ndcg_total=record.ndcg_total,
    )
    header = [
        f"- Model: `{params.embed_model}` (dim {params.embed_dimensions})",
        f"- Pipeline: `{params.pipeline}`",
        f"- Profile: `{record.profile}` (tag `{record.tag}`)",
        f"- Config generation: {params.generation}",
        f"- Run ID: `{record.run_id}`",
    ]
    view = _ReportView(
        totals=totals,
        by_category=_by_category_from_record(record.by_category),
        details=[_DetailRow.from_dict(d) for d in record.per_query],
        index_reused=record.index_reused,
        uploaded=record.uploaded,
        skipped_reasons=list(record.skipped_reasons),
        peak_rss_mb=record.peak_rss_mb,
        wall_seconds=record.wall_seconds,
        latency_p50_ms=record.latency_p50_ms,
        latency_p95_ms=record.latency_p95_ms,
        base_url=record.base_url,
        repo_root=record.repo_root,
        repo_commit=record.repo_commit,
        queries_path=record.queries_path,
        date_str=record.created_at,
        extra_header_lines=header,
        repo_dirty=record.repo_dirty,
    )
    return _render(view)


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
    """从 EvaluationRun 渲染 markdown（ad-hoc / 测试入口）。

    与 render_record 共用 _render，故输出结构一致。sweep 的落盘路径走 render_record
    （从 RunRecord，使 md 与 json 同源）；此入口供尚无 record 的临时渲染与回归测试。

    Args:
        run: harness.evaluate / run_queries 的产物。
        base_url / repo_root / queries_path: 报告头部的溯源信息。
        repo_commit: 被测仓库 HEAD（调用方取好传入；None 显示 "uncommitted"）。
        date_str: 报告日期（调用方用 ``datetime.now(timezone.utc)`` 传入，避免此纯函数
            依赖时钟；空则不显示 Date 行）。
        extra_header_lines: 额外头部行 —— 注入 model / pipeline / profile / generation。
    """
    rows = run.rows
    view = _ReportView(
        totals=compute_totals(rows),
        by_category=compute_by_category(rows),
        details=[_DetailRow.from_row(r) for r in rows],
        index_reused=run.index.reused,
        uploaded=run.index.uploaded,
        skipped_reasons=list(run.index.skipped),
        peak_rss_mb=run.peak_rss_mb,
        wall_seconds=run.wall_seconds,
        latency_p50_ms=percentile(run.client_latencies_ms, 50),
        latency_p95_ms=percentile(run.client_latencies_ms, 95),
        base_url=base_url,
        repo_root=str(repo_root),
        repo_commit=repo_commit,
        queries_path=str(queries_path) if queries_path is not None else None,
        date_str=date_str,
        extra_header_lines=extra_header_lines,
    )
    return _render(view)


def write_report(output: Path, markdown: str) -> None:
    """把渲染好的 markdown 落盘（建父目录）。落盘与渲染分离，便于测试只验渲染。"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8")
