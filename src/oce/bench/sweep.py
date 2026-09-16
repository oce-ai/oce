"""参数扫描：一次索引，扫 N 组 L0 参数 —— 本合并的核心价值。

旧痛点：改一个查询期参数（top_k / 开关）都要重启服务 + 重嵌入整仓（cc-switch ~1030 blob，
几十分钟）。sweep 把"索引"与"评分"解耦：

    index once  ──►  for each param set:  reconfigure(L0 热改, 秒级) ──► run queries ──► record

热改走 Commit 2/3 的 reconfigure 端点，read-after-write 已保证"generation 前进且 effective
⊇ patch"才开跑这组查询（client.reconfigure 内部断言），杜绝"以为改了其实没改"。每组产出一份
RunRecord（含完整参数快照 + 分数 + 逐题明细），compare.py 读它出矩阵。

参数集来自两处，都归一成 ParamSet（四组 patch：retrieval/flags/milvus/rerank）：
- 声明式矩阵 TOML（``[[set]]`` + ``[set.retrieval]`` 等）——sweep 命令用。
- ``--param KEY=VALUE`` ——run 命令用；KEY 经 route_param 按所属白名单分到对应组，拼错报错
  （不静默 no-op，对齐 reconfigure 的白名单理念）。

本模块只编排、只依赖 BenchClient（可注入 MockTransport 测试），**不起真服务、不碰
subprocess**——服务生命周期由 service.py / cli.py 负责。
"""

from __future__ import annotations

import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from oce.application.commands.reconfigure import (
    HOT_FLAG_FIELDS,
    HOT_MILVUS_FIELDS,
    HOT_RERANK_FIELDS,
    HOT_RETRIEVAL_FIELDS,
)
from oce.bench.client import BenchClient, ConfigSnapshot
from oce.bench.harness import (
    EvaluationRun,
    IndexOutcome,
    index_repository,
    load_queries,
    run_queries,
)
from oce.bench.report import render_record, write_report
from oce.bench.resources import ResourceMeter
from oce.bench.runrecord import (
    ParamSnapshot,
    RunRecord,
    build_run_record,
    pipeline_token,
    save_record,
)

# ParamSet 的四组 patch 名（与 client.reconfigure 的关键字参数一一对应）。
_PATCH_GROUPS: tuple[str, ...] = ("retrieval", "flags", "milvus", "rerank")


class SweepError(Exception):
    """矩阵解析 / 参数路由 / 扫描执行失败。消息面向用户。"""


# ---------------------------------------------------------------------------
# --param KEY=VALUE 路由：按 Commit 2 白名单把 key 分到对应 patch 组
# ---------------------------------------------------------------------------


def route_param(key: str) -> str:
    """把一个 L0 参数名路由到它所属的 patch 组（retrieval/flags/milvus/rerank）。

    拼错或不可热改的 key 直接报错——绝不静默忽略（否则 sweep 会拿"以为改了"的旧配置
    跑出一组假数据）。不可热改清单（chunk_size / HNSW M / efConstruction / dense_dim）属
    L1/L2，要 reindex 或 reset，不在 L0 热改范畴。
    """
    if key in HOT_RETRIEVAL_FIELDS:
        return "retrieval"
    if key in HOT_FLAG_FIELDS:
        return "flags"
    if key in HOT_MILVUS_FIELDS:
        return "milvus"
    if key in HOT_RERANK_FIELDS:
        return "rerank"
    known = sorted(
        HOT_RETRIEVAL_FIELDS | set(HOT_FLAG_FIELDS) | HOT_MILVUS_FIELDS | HOT_RERANK_FIELDS
    )
    raise SweepError(
        f"unknown or non-hot-swappable param '{key}'; L0 hot-swappable keys: {known}"
    )


def parse_param_args(pairs: Sequence[str]) -> dict[str, dict[str, str]]:
    """把 ``["KEY=VALUE", ...]`` 解析成按组分桶的 patch dict。

    值统一按字符串收集——交给服务端 pydantic 强转（实测 '30'->30、'false'->False、
    '0.5'->0.5 都正确），客户端不猜类型。空值 ``KEY=`` 视为空字符串（布尔/数值会因
    强转失败被服务端 422 挡下，符合预期）。
    """
    grouped: dict[str, dict[str, str]] = {g: {} for g in _PATCH_GROUPS}
    for pair in pairs:
        if "=" not in pair:
            raise SweepError(f"--param expects KEY=VALUE, got '{pair}'")
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise SweepError(f"--param has empty key: '{pair}'")
        group = route_param(key)
        grouped[group][key] = value
    # 丢掉空组，保持 patch 干净（reconfigure 对空 dict 是 no-op，但少传更清晰）
    return {g: kv for g, kv in grouped.items() if kv}


# ---------------------------------------------------------------------------
# 矩阵 TOML
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSet:
    """一组 L0 参数 patch（retrieval/flags/milvus/rerank），外加一个可读 name。"""

    name: str
    retrieval: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, Any] = field(default_factory=dict)
    milvus: dict[str, Any] = field(default_factory=dict)
    rerank: dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return not (self.retrieval or self.flags or self.milvus or self.rerank)

    def patch_groups(self) -> dict[str, dict[str, Any]]:
        """非空组的 patch（供 client.reconfigure）。"""
        out = {
            "retrieval": self.retrieval,
            "flags": self.flags,
            "milvus": self.milvus,
            "rerank": self.rerank,
        }
        return {g: v for g, v in out.items() if v}


def _label_from_patch(patch_groups: dict[str, dict[str, Any]], index: int) -> str:
    """无名 param set 时，从内容派生一个可读标签：set2[top_k=30,rerank=on]。"""
    parts: list[str] = []
    for group in _PATCH_GROUPS:
        for key, value in sorted(patch_groups.get(group, {}).items()):
            parts.append(f"{key}={value}")
    body = ",".join(parts) if parts else "current"
    return f"set{index}[{body}]"


def load_matrix(path: str | Path) -> list[ParamSet]:
    """加载声明式矩阵 TOML（``[[set]]`` 数组，每个 set 可含四组子表）。

    校验：① 顶层只认 ``[[set]]`` ② 每个 set 的子表名必须在 _PATCH_GROUPS 内 ③ 子表里的
    key 必须可热改（route_param 校验）—— 拼错早发现，不让 sweep 拿假配置跑出假数据。
    name 缺省时从内容派生。空矩阵报错（无意义）。
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise SweepError(f"matrix file not found: {path}")
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    unknown_top = set(raw) - {"set"}
    if unknown_top:
        raise SweepError(
            f"{path}: unknown top-level key(s) {sorted(unknown_top)}; expected only [[set]]"
        )
    sets = raw.get("set", [])
    if not isinstance(sets, list) or not sets:
        raise SweepError(f"{path}: matrix must contain at least one [[set]]")

    result: list[ParamSet] = []
    for index, entry in enumerate(sets, 1):
        if not isinstance(entry, dict):
            raise SweepError(f"{path}: [[set]] #{index} must be a table")
        unknown_sub = set(entry) - set(_PATCH_GROUPS) - {"name"}
        if unknown_sub:
            raise SweepError(
                f"{path}: [[set]] #{index} has unknown group(s) {sorted(unknown_sub)}; "
                f"expected {list(_PATCH_GROUPS)}"
            )
        groups: dict[str, dict[str, Any]] = {}
        for group in _PATCH_GROUPS:
            sub = entry.get(group, {})
            if not isinstance(sub, dict):
                raise SweepError(f"{path}: [[set]] #{index} [{group}] must be a table")
            # 逐 key 校验可热改性（拼错/越层早报错）
            for key in sub:
                try:
                    routed = route_param(key)
                except SweepError as exc:
                    raise SweepError(
                        f"{path}: [[set]] #{index} [{group}] {exc}"
                    ) from exc
                if routed != group:
                    raise SweepError(
                        f"{path}: [[set]] #{index} key '{key}' belongs to [{routed}], "
                        f"not [{group}]"
                    )
            groups[group] = dict(sub)
        name = entry.get("name") or _label_from_patch(groups, index)
        result.append(ParamSet(name=str(name), **groups))
    return result


def param_set_from_args(pairs: Sequence[str], name: str = "adhoc") -> ParamSet:
    """把 ``--param KEY=VALUE`` 列表封成单个 ParamSet（run 命令用）。"""
    grouped = parse_param_args(pairs)
    return ParamSet(
        name=name,
        retrieval=grouped.get("retrieval", {}),
        flags=grouped.get("flags", {}),
        milvus=grouped.get("milvus", {}),
        rerank=grouped.get("rerank", {}),
    )


# ---------------------------------------------------------------------------
# 参数快照：从 reconfigure ack 的 effective 构造 ParamSnapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepContext:
    """一次 sweep/run 期间**恒定**的溯源元数据（每组参数共享，故抽出来避免签名爆炸）。

    这些值不随 param set 变化：目标仓库、被测模型/维度/后端、profile/tag、HNSW 建库参数。
    L0 effective（随热改变化）不在此——它在每组循环里从 get_config 现取。
    """

    repo_name: str
    repo_root: Path
    repo_commit: str | None
    repo_dirty: bool
    queries_path: Path
    profile_name: str
    tag: str
    embed_model: str
    embed_dimensions: int
    embed_endpoint: str
    db_dialect: str
    milvus_mode: str
    hnsw_m: int | None = None
    hnsw_ef_construction: int | None = None

    def snapshot(self, config: ConfigSnapshot) -> ParamSnapshot:
        """把恒定的 L2/L1 溯源 + 本轮 get_config 的 L0 effective 合成 ParamSnapshot。

        effective 是"这组分数确切跑在哪套参数下"的唯一真源；pipeline token 由其 flags 派生。
        """
        effective = config.effective
        flags = dict(effective.get("flags", {}))
        return ParamSnapshot(
            embed_model=self.embed_model,
            embed_dimensions=self.embed_dimensions,
            embed_endpoint=self.embed_endpoint,
            db_dialect=self.db_dialect,
            milvus_mode=self.milvus_mode,
            hnsw_m=self.hnsw_m,
            hnsw_ef_construction=self.hnsw_ef_construction,
            generation=config.generation,
            effective=effective,
            pipeline=pipeline_token(flags),
        )


# ---------------------------------------------------------------------------
# 扫描主循环
# ---------------------------------------------------------------------------


async def run_sweep(
    client: BenchClient,
    context: SweepContext,
    *,
    param_sets: Sequence[ParamSet],
    runs_dir: Path,
    reuse_index: bool = False,
    embedding_timeout: float = 3600.0,
    concurrency: int = 1,
    created_at_provider: Callable[[], Any] | None = None,
    meter: ResourceMeter | None = None,
    log: Callable[[str], None] | None = None,
) -> list[RunRecord]:
    """索引一次，对每组参数热改 + 跑查询 + 落 RunRecord（JSON + MD）。返回全部记录。

    流程：
    1. ``index_repository`` 一次（reuse_index 控制新上传还是复用既有索引）——最贵的动作只此一次。
    2. 对每个 ParamSet：
       a. 非空则 ``client.reconfigure``（read-after-write 校验 generation 前进 + effective⊇patch）；
          空集则用当前配置（只 GET 一次拿 effective）。
       b. ``run_queries``（**不重新索引**，复用上一步的 scope）。
       c. ``build_run_record`` + ``save_record`` 落 JSON，``render_record`` + ``write_report``
          从该 record 落同源 MD（头部注入 model/pipeline/profile/generation）。
    3. 返回全部 RunRecord（compare 直接吃）。

    created_at_provider 便于测试注入固定时钟（默认 datetime.now(timezone.utc)，AGENTS.md
    禁 utcnow）。meter 为 None 时不采内存。
    """
    emit = log or (lambda _msg: None)

    if not param_sets:
        raise SweepError("run_sweep requires at least one param set")

    # --- 索引一次（最贵的动作，全流程只发生一次）---
    emit(f"indexing {context.repo_name} (reuse_index={reuse_index})")
    index: IndexOutcome = await index_repository(
        client,
        context.repo_root,
        reuse_index=reuse_index,
        embedding_timeout=embedding_timeout,
        meter=meter,
        log=log,
    )
    scope = index.blob_names
    queries = load_queries(context.queries_path)
    emit(f"indexed {index.uploaded} blobs; loaded {len(queries)} queries")

    records: list[RunRecord] = []
    for set_index, param_set in enumerate(param_sets, 1):
        emit(f"=== [{set_index}/{len(param_sets)}] {param_set.name} ===")

        # --- L0 热改（read-after-write 校验）或读当前配置 ---
        if param_set.is_empty():
            emit("no param patch; using current config")
            config = await client.get_config()
        else:
            groups = param_set.patch_groups()
            ack = await client.reconfigure(
                retrieval=groups.get("retrieval") or None,
                flags=groups.get("flags") or None,
                milvus=groups.get("milvus") or None,
                rerank=groups.get("rerank") or None,
            )
            emit(
                f"reconfigured -> generation={ack.generation} "
                f"reranker_reloaded={ack.reranker_reloaded}"
            )
            # reconfigure 内部已 GET 校验；再取一次 config 作为快照来源（与 ack 一致）
            config = await client.get_config()

        params = context.snapshot(config)

        # --- 跑查询（复用索引，不重嵌入）---
        set_started = time.monotonic()
        rows, latencies = await run_queries(
            client,
            queries,
            scope,
            concurrency=concurrency,
            meter=meter,
            log=log,
        )
        set_wall = time.monotonic() - set_started
        run = EvaluationRun(
            rows=rows,
            index=index,
            peak_rss_mb=meter.peak_rss_mb() if meter else 0.0,
            wall_seconds=set_wall,
            client_latencies_ms=latencies,
        )

        # created_at 只取一次：JSON 的 run_id 与 MD 文件名据此配对，二者永不漂移
        created_at = created_at_provider() if created_at_provider else None
        record = build_run_record(
            run=run,
            params=params,
            repo_name=context.repo_name,
            repo_root=context.repo_root,
            repo_commit=context.repo_commit,
            repo_dirty=context.repo_dirty,
            queries_path=context.queries_path,
            profile=context.profile_name,
            tag=context.tag,
            base_url=str(client.base_url),
            created_at=created_at,
        )
        json_path = save_record(record, runs_dir)

        # MD 人读视图：与 JSON **同源**（render_record 直接吃刚落盘的 RunRecord），头部自动
        # 注入 model/pipeline/profile/generation（补上旧报告不记录用了哪个模型的缺口）。这条
        # 路径与离线重渲（oce bench report 读 JSON）走的是同一个函数，故 md 与 json 永不漂移。
        md_path = runs_dir / record.md_filename
        markdown = render_record(record)
        write_report(md_path, markdown)

        emit(
            f"score={record.score_pct:.1f}% top1={record.top1_total}/"
            f"{record.query_count} ndcg_sum={record.ndcg_total:.2f} -> {json_path.name}"
        )
        records.append(record)

    emit(f"sweep complete: {len(records)} runs -> {runs_dir}")
    return records
