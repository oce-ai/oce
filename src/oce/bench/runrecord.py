"""结构化 run 记录：每次评测的完整参数快照 + 分数 + 逐题明细，落成 JSON。

这是"对比变成数据查询而非正则刮 markdown"的基石。旧机制靠文件名约定
（``{model}__{pipeline}__[repo]__d{dim}__{date}.md``）承载参数，再用正则从报告里反向刮分数
——脆弱、易漂移、附录里一堆"缩写语义待确认"。RunRecord 把参数与分数一起结构化，compare.py
直接读 JSON 出矩阵，每列自动标注改了哪个参数。

**分层快照**：ParamSnapshot 记录 L2（模型/维度/后端）、L1（HNSW 建库参数）、L0（热改后的
effective 全集 + generation + pipeline token）。L0 的 effective 取自 reconfigure 之后
get_config 的返回——即"这组分数是在这套确切生效参数下跑出来的"，read-after-write 已保证。

JSON 是 compare 的唯一真源；md 是人读视图，由 report.py 从**同一次** EvaluationRun 渲染
（头部注入 model/pipeline/profile/generation，补上旧报告不记录用了哪个模型的缺口）。二者
同源，Commit 6 内不漂移；Commit 7 进一步让 md 从 RunRecord 渲染，使 .json 可离线重渲。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from oce.bench.harness import EvaluationRow, EvaluationRun
from oce.bench.report import compute_by_category, compute_totals, percentile


def pipeline_token(flags: dict[str, Any]) -> str:
    """从生效 flags 派生 pipeline 短标签（base/rerank/llmrerank/llmintent）。

    按"最高启用层"命名（与旧文件名约定的 pipeline token 一致，kNN intent 已按决定丢弃）：
    LLM 语义重排 > API 重排 > LLM 意图分类 > base。flags 全集另存于 ParamSnapshot.effective，
    此 token 只是给人读/排序用的粗标签，不是唯一真源。
    """
    if flags.get("llm_rerank_enabled"):
        return "llmrerank"
    if flags.get("rerank_enabled"):
        return "rerank"
    if flags.get("intent_classification_enabled"):
        return "llmintent"
    return "base"


@dataclass
class ParamSnapshot:
    """一次 run 的完整生效参数（L2/L1/L0 分层）。"""

    # --- L2：换它要完整 reset + 重启 + 重嵌入 ---
    embed_model: str = ""
    embed_dimensions: int = 0
    embed_endpoint: str = ""
    db_dialect: str = ""
    milvus_mode: str = ""

    # --- L1：换它要 drop collection + reindex（服务不重启）---
    hnsw_m: int | None = None
    hnsw_ef_construction: int | None = None

    # --- L0：查询期热改（秒级，见 reconfigure）---
    generation: int = 0
    # reconfigure 之后 get_config 的 effective 全集：retrieval/flags/milvus/rerank
    effective: dict[str, Any] = field(default_factory=dict)
    # 由 effective["flags"] 派生的粗标签（base/rerank/llmrerank/llmintent）
    pipeline: str = "base"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ParamSnapshot:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RunRecord:
    """一次评测的完整结构化记录（JSON 落盘的单位）。"""

    run_id: str
    created_at: str  # ISO-8601 UTC（AGENTS.md 禁 utcnow，用 now(timezone.utc)）

    # --- 目标仓库溯源 ---
    repo_name: str
    repo_root: str
    repo_commit: str | None  # 被测仓库 HEAD；None = 未提交/取不到
    repo_dirty: bool  # 工作区是否有未提交改动（lock 状态：dirty 的分数复现性存疑）
    queries_path: str
    query_count: int

    # --- 参数 + 环境溯源 ---
    params: ParamSnapshot
    profile: str
    tag: str
    base_url: str

    # --- 聚合分数 ---
    score_pct: float
    earned: float
    max_points: float
    top1_total: int
    ndcg_total: float
    by_category: dict[str, list[float]]  # cat -> [count, top1, ndcg_sum]
    by_difficulty: dict[str, list[float]]  # difficulty -> [count, top1, ndcg_sum]

    # --- 逐题明细 ---
    per_query: list[dict[str, Any]]

    # --- 资源 / 时延 ---
    peak_rss_mb: float
    wall_seconds: float
    latency_p50_ms: float
    latency_p95_ms: float
    index_reused: bool
    uploaded: int
    skipped: int

    # --- token 用量（monitoring 开时从 /admin/reports/tokens 拉；Commit 7 接线）---
    tokens: dict[str, Any] | None = None

    # --- 同目录 md 文件名（人读视图）---
    md_filename: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        payload = dict(data)
        payload["params"] = ParamSnapshot.from_dict(payload.get("params", {}))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


# ---------------------------------------------------------------------------
# 逐题明细 / 难度聚合
# ---------------------------------------------------------------------------


def _row_to_dict(row: EvaluationRow) -> dict[str, Any]:
    """EvaluationRow -> 精简 dict（不存 formatted 全文，太大；存路径与分数即可复盘）。"""
    return {
        "query_id": row.query_id,
        "category": row.category,
        "difficulty": row.difficulty,
        "expected_files": list(row.expected_files),
        "top_paths": list(row.top_paths),
        "top1_score": row.top1_score,
        "ndcg_score": row.ndcg_score,
        "total": row.total,
        "client_elapsed_ms": row.client_elapsed_ms,
        "server_elapsed_ms": row.server_elapsed_ms,
        "error": row.error,
    }


def compute_by_difficulty(
    rows: Sequence[EvaluationRow],
) -> dict[str, list[float]]:
    """按难度聚合：``difficulty -> [题数, top1 命中数, ndcg 累加]``（键按数值排序）。"""
    buckets: dict[int, list[float]] = {}
    for row in rows:
        slot = buckets.setdefault(row.difficulty, [0, 0, 0.0])
        slot[0] += 1
        slot[1] += row.top1_score
        slot[2] += row.ndcg_score
    return {str(k): buckets[k] for k in sorted(buckets)}


def _by_category_serializable(
    by_category: dict[str, tuple[int, int, float]],
) -> dict[str, list[float]]:
    return {cat: [float(c), float(t), float(n)] for cat, (c, t, n) in by_category.items()}


# ---------------------------------------------------------------------------
# 构造 + 持久化
# ---------------------------------------------------------------------------


def make_run_id(
    *,
    created_at: datetime,
    repo_name: str,
    pipeline: str,
    embed_model: str,
    dimensions: int,
    tag: str,
    generation: int = 0,
) -> str:
    """自描述、可按时间排序的 run_id。

    ``{UTC 时间戳}__{repo}__{pipeline}__{model}__d{dim}__g{generation}__{tag}`` —— 时间戳
    保证排序，``generation`` 保证**同一次 sweep 内多组参数不撞名**（秒级时间戳 + 仓库 +
    pipeline + 模型 + 维度 + tag 在一次扫描里全恒定，缺了 generation 两组就会同名静默覆盖；
    每次 reconfigure 都让 generation 前进，故各组天然可区分）。其余段让人一眼看出这条 run
    测了什么（对齐旧文件名约定，但结构化字段才是真源）。
    """
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    # model 可能含 / : 等不适合文件名的字符，压平成 -
    safe_model = embed_model.replace("/", "-").replace(":", "-")
    return (
        f"{stamp}__{repo_name}__{pipeline}__{safe_model}__d{dimensions}"
        f"__g{generation}__{tag}"
    )


def build_run_record(
    *,
    run: EvaluationRun,
    params: ParamSnapshot,
    repo_name: str,
    repo_root: Path,
    repo_commit: str | None,
    repo_dirty: bool,
    queries_path: Path,
    profile: str,
    tag: str,
    base_url: str,
    created_at: datetime | None = None,
    tokens: dict[str, Any] | None = None,
) -> RunRecord:
    """从一次 EvaluationRun + 参数快照组装完整 RunRecord（聚合分数在此算好）。"""
    created = created_at or datetime.now(timezone.utc)
    rows = run.rows
    totals = compute_totals(rows)
    by_category = compute_by_category(rows)

    run_id = make_run_id(
        created_at=created,
        repo_name=repo_name,
        pipeline=params.pipeline,
        embed_model=params.embed_model,
        dimensions=params.embed_dimensions,
        tag=tag,
        generation=params.generation,
    )
    return RunRecord(
        run_id=run_id,
        created_at=created.isoformat(),
        repo_name=repo_name,
        repo_root=str(repo_root),
        repo_commit=repo_commit,
        repo_dirty=repo_dirty,
        queries_path=str(queries_path),
        query_count=totals.query_count,
        params=params,
        profile=profile,
        tag=tag,
        base_url=base_url,
        score_pct=totals.score_pct,
        earned=totals.earned,
        max_points=totals.max_points,
        top1_total=totals.top1_total,
        ndcg_total=totals.ndcg_total,
        by_category=_by_category_serializable(by_category),
        by_difficulty=compute_by_difficulty(rows),
        per_query=[_row_to_dict(row) for row in rows],
        peak_rss_mb=run.peak_rss_mb,
        wall_seconds=run.wall_seconds,
        latency_p50_ms=percentile(run.client_latencies_ms, 50),
        latency_p95_ms=percentile(run.client_latencies_ms, 95),
        index_reused=run.index.reused,
        uploaded=run.index.uploaded,
        skipped=len(run.index.skipped),
        tokens=tokens,
        md_filename=f"{run_id}.md",
    )


def save_record(record: RunRecord, runs_dir: Path) -> Path:
    """把 RunRecord 写成 ``<runs_dir>/<run_id>.json``（建父目录，UTF-8，缩进 2）。

    返回写入路径。md 由调用方（cli/sweep）用 report.py 从同一 EvaluationRun 渲染并落盘到
    同目录 ``<run_id>.md``——record.md_filename 已记好该名字，二者配对。
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{record.run_id}.json"
    path.write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_record(path: str | Path) -> RunRecord:
    """从 JSON 读回 RunRecord（compare 的入口）。"""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        return RunRecord.from_dict(json.load(handle))


def list_records(runs_dir: str | Path) -> list[RunRecord]:
    """列出目录下所有 ``*.json`` run 记录，按 run_id（含时间戳前缀）排序。"""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    records = [load_record(p) for p in runs_dir.glob("*.json")]
    return sorted(records, key=lambda r: r.run_id)
