"""评测编排：索引一个仓库 + 跑一个查询集 → 逐题行 + 资源指标。

harness 只负责"跑一次评测"的流程；评分口径在 scoring.py，传输在 client.py，资源采样在
resources.py。产出 ``EvaluationRow`` 列表交给 report.py 渲染、交给 runrecord.py 落结构化记录。

从 oce-benchmark/scripts/run_retrieval_eval.py 的 evaluate_one / run 移植，三处修正：
1. 复用 client.py 的传输层（不再内联 httpx 调用与端点路径）。
2. 查询循环保持**串行**（确定性优先：并发会让时延指标失真、让 nDCG 受调度抖动影响），
   但抽出 ``concurrency`` 参数留作可选加速（默认 1）。
3. RSS 测量交给 resources.py（psutil），不再用 Windows 专用的 ctypes psapi。
"""

from __future__ import annotations

import asyncio
import gc
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from oce.bench.blobs import SourceBlob, iter_source_blobs, make_batches
from oce.bench.client import BenchClient, RetrievalOutcome
from oce.bench.resources import ResourceMeter
from oce.bench.scoring import score_query


@dataclass
class EvaluationRow:
    """单个查询的评测结果行（评分 + 明细，报告与 run 记录共用）。"""

    query_id: str
    category: str
    difficulty: int
    query: str
    expected_files: list[str]
    top_paths: list[str]
    formatted: str
    client_elapsed_ms: int
    server_elapsed_ms: int
    top1_score: int
    ndcg_score: float
    error: str | None = None

    @property
    def total(self) -> float:
        """本题得分（满分 2：Top-1 1 分 + nDCG@10 至多 1 分）。"""
        return self.top1_score + self.ndcg_score


@dataclass(frozen=True)
class IndexOutcome:
    """索引阶段的产物：scope（检索工作集）+ 上传统计 + 是否复用。"""

    blob_names: list[str]
    uploaded: int
    skipped: list[str]
    reused: bool


@dataclass
class EvaluationRun:
    """一次完整评测（索引 + 查询）的结果。"""

    rows: list[EvaluationRow]
    index: IndexOutcome
    peak_rss_mb: float
    wall_seconds: float
    client_latencies_ms: list[int] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for row in self.rows if row.error)


def load_queries(path: Path) -> list[dict]:
    """读 JSONL 查询集（容忍 utf-8-sig BOM 与空行）。

    一次性读成 list（不再像 harness 那样惰性两次遍历）：查询集很小（100 题），且 run 记录
    需要总数先行确定。
    """
    import json

    queries: list[dict] = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                queries.append(json.loads(line))
    return queries


async def index_repository(
    client: BenchClient,
    repo_root: Path,
    *,
    reuse_index: bool,
    embedding_timeout: float,
    meter: ResourceMeter | None = None,
    log: Callable[[str], None] | None = None,
) -> IndexOutcome:
    """索引一个仓库，返回检索工作集（blob_names）。

    ``reuse_index=True``：本地重算全部 blob_name → /find-missing 查服务端状态 →
    有 unknown 直接报错（需先不带 --reuse-index 上传）→ 有 nonindexed 则等嵌入。
    ``reuse_index=False``：完整上传（poison-split 兜底）→ 等全部嵌入完成。

    两条路径都产出同一份 ``blob_names`` 作为检索 scope —— 模拟真实客户端保存 blob 列表、
    检索时用 added_blobs 传递。``log`` 报告编排层进度（传输层进度由 client 自己报告）。
    """
    emit = log or (lambda _msg: None)
    skipped: list[str] = []
    names: list[str] = []

    if reuse_index:
        emit("computing blob names from local files")
        for blob in iter_source_blobs(repo_root):
            names.append(blob.blob_name)
        uploaded = len(names)
        if meter:
            meter.sample()
        emit(f"computed {uploaded} blob names")

        emit("checking server-side blob status")
        unknown, nonindexed = await client.find_missing(names)
        if unknown:
            raise RuntimeError(
                f"{len(unknown)} blobs not found on server; run without --reuse-index "
                f"to upload them first (e.g. {unknown[0][:16]})"
            )
        if nonindexed:
            emit(f"waiting for {len(nonindexed)} blobs to be indexed")
            await client.wait_for_embedding(nonindexed, timeout=embedding_timeout)
        else:
            emit("all blobs already indexed")
        return IndexOutcome(
            blob_names=names, uploaded=uploaded, skipped=skipped, reused=True
        )

    # 完整上传
    emit("uploading repository files")
    uploaded = 0
    for batch_number, batch in enumerate(
        make_batches(iter_source_blobs(repo_root)), 1
    ):
        batch_names, batch_skipped = await client.upload_batch(batch)
        names.extend(batch_names)
        skipped.extend(batch_skipped)
        uploaded = len(names)
        if meter:
            meter.sample()
        if batch_number % 10 == 0:
            emit(f"batch={batch_number} blobs={uploaded}")
        # 主动释放批次引用并 GC：大仓上传时避免内存单调增长（harness 原有行为）
        del batch
        gc.collect()

    emit("waiting for embedding to complete")
    await client.wait_for_embedding(names, timeout=embedding_timeout)
    return IndexOutcome(
        blob_names=names, uploaded=uploaded, skipped=skipped, reused=False
    )


async def _run_one_query(
    client: BenchClient,
    query: dict,
    scope: Sequence[str],
) -> EvaluationRow:
    """跑单个查询、评分、产出 EvaluationRow。查询失败记 error 行（不中断整轮）。"""
    qid = query.get("query_id") or query.get("id") or "?"
    category = query.get("category", "unknown")
    difficulty = query.get("difficulty", 1)
    text = query["query"]
    expected = query["expected_files"]

    try:
        outcome: RetrievalOutcome = await client.retrieve(text, added_blobs=scope)
    except Exception as exc:  # 单题失败不中断：记 error 行，继续
        return EvaluationRow(
            query_id=qid,
            category=category,
            difficulty=difficulty,
            query=text,
            expected_files=list(expected),
            top_paths=[],
            formatted="",
            client_elapsed_ms=0,
            server_elapsed_ms=0,
            top1_score=0,
            ndcg_score=0.0,
            error=str(exc),
        )

    top_paths, top1, ndcg = score_query(query, outcome.formatted)
    return EvaluationRow(
        query_id=qid,
        category=category,
        difficulty=difficulty,
        query=text,
        expected_files=list(expected),
        top_paths=top_paths,
        formatted=outcome.formatted,
        client_elapsed_ms=outcome.client_elapsed_ms,
        server_elapsed_ms=outcome.server_elapsed_ms,
        top1_score=top1,
        ndcg_score=ndcg,
    )


async def run_queries(
    client: BenchClient,
    queries: Sequence[dict],
    scope: Sequence[str],
    *,
    concurrency: int = 1,
    meter: ResourceMeter | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[list[EvaluationRow], list[int]]:
    """跑全部查询，返回 ``(rows, client_latencies_ms)``。

    默认串行（concurrency=1）：确定性优先。``concurrency>1`` 时用信号量限流并发，仅用于
    追求吞吐、不在意时延指标精度的场景。
    """
    emit = log or (lambda _msg: None)
    rows: list[EvaluationRow] = []
    latencies: list[int] = []

    if concurrency <= 1:
        for query in queries:
            row = await _run_one_query(client, query, scope)
            rows.append(row)
            latencies.append(row.client_elapsed_ms)
            if meter:
                meter.sample()
            emit(_query_progress(row))
        return rows, latencies

    semaphore = asyncio.Semaphore(concurrency)

    async def guarded(query: dict) -> EvaluationRow:
        async with semaphore:
            return await _run_one_query(client, query, scope)

    results = await asyncio.gather(*(guarded(q) for q in queries))
    # 并发下按输入顺序回填，保持确定性输出；latency 取每题客户端 wall time
    for row in results:
        rows.append(row)
        latencies.append(row.client_elapsed_ms)
        emit(_query_progress(row))
    if meter:
        meter.sample()
    return rows, latencies


def _query_progress(row: EvaluationRow) -> str:
    if row.error:
        return f"x {row.query_id}: ERROR - {row.error}"
    status = "ok" if (row.top1_score or row.ndcg_score) else "miss"
    return f"{status} {row.query_id}: top1={row.top1_score} ndcg={row.ndcg_score:.3f}"


async def evaluate(
    client: BenchClient,
    repo_root: Path,
    queries_path: Path,
    *,
    reuse_index: bool = False,
    embedding_timeout: float = 3600.0,
    concurrency: int = 1,
    meter: ResourceMeter | None = None,
    log: Callable[[str], None] | None = None,
) -> EvaluationRun:
    """一次完整评测：索引仓库 → 跑查询集 → 收集资源指标。

    harness 的顶层入口。cli.run / sweep 都调它；它不做参数解析、不落盘、不渲染，
    只产出结构化 EvaluationRun 交调用方处置。``log`` 报告编排层进度。
    """
    emit = log or (lambda _msg: None)
    started = time.monotonic()
    index = await index_repository(
        client,
        repo_root,
        reuse_index=reuse_index,
        embedding_timeout=embedding_timeout,
        meter=meter,
        log=log,
    )

    queries = load_queries(queries_path)
    emit(f"running {len(queries)} queries")
    rows, latencies = await run_queries(
        client,
        queries,
        index.blob_names,
        concurrency=concurrency,
        meter=meter,
        log=log,
    )
    wall_seconds = time.monotonic() - started

    peak_rss_mb = meter.peak_rss_mb() if meter else 0.0
    emit(
        f"evaluation complete: {len(rows)} queries, "
        f"{sum(1 for r in rows if r.top1_score)} top-1 hits, "
        f"peak_rss={peak_rss_mb:.1f}MB, wall={wall_seconds:.1f}s"
    )
    return EvaluationRun(
        rows=rows,
        index=index,
        peak_rss_mb=peak_rss_mb,
        wall_seconds=wall_seconds,
        client_latencies_ms=latencies,
    )
