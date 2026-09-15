"""报告渲染测试（纯函数，用手工 EvaluationRun，不起服务）。

覆盖：聚合分数、按分类汇总、百分位、以及 Commit 4 的两处改进——解耦 git（repo_commit
传参）与补 p50/p95 时延。extra_header_lines 是 Commit 7 注入 model/pipeline/generation 的
预留口，此处一并锁定。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oce.bench.harness import EvaluationRow, EvaluationRun, IndexOutcome
from oce.bench.report import (
    compute_by_category,
    compute_totals,
    percentile,
    render_report,
    write_report,
)


def _row(
    qid: str,
    category: str,
    *,
    top1: int,
    ndcg: float,
    difficulty: int = 1,
    client_ms: int = 100,
    server_ms: int = 80,
    error: str | None = None,
    top_paths: list[str] | None = None,
) -> EvaluationRow:
    return EvaluationRow(
        query_id=qid,
        category=category,
        difficulty=difficulty,
        query=f"question {qid}",
        expected_files=["a.py"],
        top_paths=top_paths or (["a.py"] if top1 else []),
        formatted="",
        client_elapsed_ms=client_ms,
        server_elapsed_ms=server_ms,
        top1_score=top1,
        ndcg_score=ndcg,
        error=error,
    )


def _run(rows: list[EvaluationRow], **kwargs) -> EvaluationRun:
    return EvaluationRun(
        rows=rows,
        index=IndexOutcome(
            blob_names=["n1", "n2"], uploaded=2, skipped=[], reused=kwargs.get("reused", False)
        ),
        peak_rss_mb=kwargs.get("peak_rss_mb", 123.4),
        wall_seconds=kwargs.get("wall_seconds", 5.0),
        client_latencies_ms=kwargs.get("latencies", [r.client_elapsed_ms for r in rows]),
    )


class TestPercentile:
    def test_empty(self):
        assert percentile([], 50) == 0.0

    def test_single(self):
        assert percentile([42], 95) == 42.0

    def test_median_nearest_rank(self):
        assert percentile([10, 20, 30, 40, 50], 50) == 30.0

    def test_p95_of_hundred(self):
        values = list(range(1, 101))
        assert percentile(values, 95) == 95.0

    def test_unsorted_input(self):
        assert percentile([50, 10, 30, 20, 40], 50) == 30.0


class TestTotals:
    def test_aggregate(self):
        rows = [
            _row("Q1", "cat_a", top1=1, ndcg=1.0),
            _row("Q2", "cat_a", top1=0, ndcg=0.5),
            _row("Q3", "cat_b", top1=1, ndcg=0.0),
        ]
        totals = compute_totals(rows)
        assert totals.query_count == 3
        assert totals.top1_total == 2
        assert totals.ndcg_total == pytest.approx(1.5)
        assert totals.earned == pytest.approx(3.5)
        assert totals.max_points == 6  # 3 * POINTS_PER_QUERY
        assert totals.score_pct == pytest.approx(3.5 / 6 * 100)

    def test_by_category(self):
        rows = [
            _row("Q1", "cat_a", top1=1, ndcg=1.0),
            _row("Q2", "cat_a", top1=0, ndcg=0.5),
            _row("Q3", "cat_b", top1=1, ndcg=0.0),
        ]
        by_cat = compute_by_category(rows)
        assert by_cat["cat_a"] == (2, 1, 1.5)  # (count, top1, ndcg_sum)
        assert by_cat["cat_b"] == (1, 1, 0.0)
        # 分类按名排序
        assert list(by_cat) == ["cat_a", "cat_b"]


class TestRenderReport:
    # _run 专用键（资源/索引指标）与 render_report 专用键（溯源/头部）分两组，
    # 避免 kwargs 串味导致 TypeError。
    _RUN_KEYS = ("reused", "peak_rss_mb", "wall_seconds", "latencies")

    def _md(
        self,
        *,
        rows: list[EvaluationRow] | None = None,
        base_url: str = "http://127.0.0.1:8987",
        repo_commit: str | None = None,
        date_str: str = "",
        extra_header_lines: list[str] | None = None,
        **run_kwargs,
    ) -> str:
        rows = rows or [
            _row("Q1", "cat_a", top1=1, ndcg=1.0, client_ms=120),
            _row("Q2", "cat_a", top1=0, ndcg=0.5, client_ms=200),
        ]
        assert all(k in self._RUN_KEYS for k in run_kwargs), run_kwargs
        return render_report(
            _run(rows, **run_kwargs),
            base_url=base_url,
            repo_root=Path("/repo/flask"),
            queries_path=Path("/bench/flask.jsonl"),
            repo_commit=repo_commit,
            date_str=date_str,
            extra_header_lines=extra_header_lines or [],
        )

    def test_contains_total_table(self):
        md = self._md()
        assert "# OCE Retrieval Evaluation" in md
        assert "## Total" in md
        assert "## Categories" in md
        assert "cat_a" in md

    def test_commit_displayed(self):
        md = self._md(repo_commit="abc123")
        assert "Repository SHA: `abc123`" in md

    def test_commit_defaults_uncommitted(self):
        md = self._md()
        assert "Repository SHA: `uncommitted`" in md

    def test_date_omitted_when_empty(self):
        md = self._md(date_str="")
        assert "- Date:" not in md

    def test_date_shown_when_given(self):
        md = self._md(date_str="2026-09-15")
        assert "- Date: 2026-09-15" in md

    def test_latency_percentiles_present(self):
        """Commit 4 新增：p50/p95 时延（原报告缺失）。"""
        md = self._md(latencies=[100, 120, 200, 300, 900])
        assert "Query latency (client): p50" in md
        assert "p95" in md

    def test_extra_header_lines_injected(self):
        """Commit 7 预留口：model/pipeline/generation 头部行。"""
        md = self._md(
            extra_header_lines=["- Model: `f2llm-v2`", "- Generation: 3"]
        )
        assert "- Model: `f2llm-v2`" in md
        assert "- Generation: 3" in md

    def test_reused_index_noted(self):
        md = self._md(reused=True)
        assert "Index: reused" in md

    def test_fresh_index_noted(self):
        md = self._md(reused=False)
        assert "Index: fresh upload" in md

    def test_details_per_query(self):
        md = self._md()
        assert "### Q1 (cat_a, difficulty 1)" in md
        assert "question Q1" in md  # query 文本入正文
        assert "Elapsed:" in md
        # Commit 4：客户端 + 服务端双视角耗时
        assert "(client)" in md and "(server)" in md

    def test_error_rows_section(self):
        rows = [
            _row("Q1", "cat_a", top1=0, ndcg=0.0, error="HTTP 500 boom"),
            _row("Q2", "cat_a", top1=1, ndcg=1.0),
        ]
        md = self._md(rows=rows)
        assert "## Query errors" in md
        assert "HTTP 500 boom" in md


class TestWriteReport:
    def test_creates_parent_dirs_and_writes(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "report.md"
        write_report(out, "# hi\n")
        assert out.is_file()
        assert out.read_text(encoding="utf-8") == "# hi\n"
