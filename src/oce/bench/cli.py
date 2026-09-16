"""``oce bench`` 子命令：一条命令链完成评测（list/serve/run/sweep/reconfigure/compare/reset）。

取代散落在 scratch 的 bench_orchestrate / serve_bench / reset_bench 三脚本，且：
- 无硬编码绝对路径（数据集从包内发现、被测仓库按 --repo-root/env/同级约定解析）。
- 无明文密码（profile 的 ``*_env`` 引用 + secrets.env，见 profiles.py）。
- 改参数秒级生效不重启（reconfigure/sweep 走 L0 热改端点，read-after-write 保证）。

命令分两类：
- **服务侧**（起/清隔离服务，需 import app、灌 env）：``serve`` / ``reset``。
- **客户端侧**（连已运行的服务，纯 httpx）：``run`` / ``sweep`` / ``reconfigure`` /
  ``compare`` / ``list``。典型流程：先 ``serve`` 起服务（长驻），另开终端 ``run``/``sweep``
  打它——同一份索引可被多组参数复用。

handler 是 argparse 的同步回调；异步编排（run/sweep/reconfigure）在 handler 内 asyncio.run。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

from oce.bench.client import BenchClient, ReconfigureRejected
from oce.bench.compare import (
    CompareError,
    find_baseline,
    load_runs,
    render_compare,
)
from oce.bench.datasets import (
    Dataset,
    DatasetError,
    discover_datasets,
    find_dataset,
    resolve_repo_root,
)
from oce.bench.profiles import (
    Profile,
    ProfileError,
    default_profiles_dir,
    load_profile,
)
from oce.bench.resources import build_meter
from oce.bench.sweep import (
    ParamSet,
    SweepContext,
    SweepError,
    load_matrix,
    param_set_from_args,
    run_sweep,
)

# bench 专属默认数据目录（与个人模式 ~/.oce/data 隔离，避免串库）
_DEFAULT_BENCH_DATA_DIR = Path.home() / ".oce" / "bench-data"
# 默认 run 产物目录（仓库根 bench/runs，.gitignore；compare 的唯一真源）
_DEFAULT_RUNS_DIR = Path("bench") / "runs"


class BenchCLIError(Exception):
    """bench 子命令的用户级错误（参数误用等）；_guard 捕获后打印并退出码 2。"""


# 用户级错误：打印 ``error: ...`` 到 stderr 并退出码 2，绝不抛栈（main oce cli 的 main()
# 没有 try/except，故 guard 必须在 handler 内消化错误再 SystemExit）。
_USER_ERRORS = (
    BenchCLIError,
    ProfileError,
    DatasetError,
    SweepError,
    CompareError,
    ReconfigureRejected,
)


def _guard(handler: Callable[[argparse.Namespace], None]):
    """把 bench handler 包成"出错即打印 + sys.exit(2)"，使其可安全挂到主 cli 的 handler 链。"""

    def wrapped(args: argparse.Namespace) -> None:
        try:
            handler(args)
        except _USER_ERRORS as exc:
            print(f"error: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc

    return wrapped


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _resolve_profile(name_or_path: str) -> Profile:
    """``--profile`` 取值：先当路径，再当 bench/profiles/<name>.toml 的短名。"""
    candidate = Path(name_or_path)
    if candidate.is_file():
        return load_profile(candidate)
    short = default_profiles_dir() / f"{name_or_path}.toml"
    if short.is_file():
        return load_profile(short)
    example = default_profiles_dir() / f"{name_or_path}.example.toml"
    if example.is_file():
        return load_profile(example)
    raise BenchCLIError(
        f"profile '{name_or_path}' not found (tried {candidate}, {short}, {example})"
    )


def _resolve_api_key(explicit: str | None) -> str:
    """API key 优先级：--api-key > 环境变量 OCE_BENCH_API_KEY > API_KEY。"""
    import os

    for value in (explicit, os.environ.get("OCE_BENCH_API_KEY"), os.environ.get("API_KEY")):
        if value:
            return value
    raise BenchCLIError(
        "no API key; pass --api-key or set OCE_BENCH_API_KEY (must match the service)"
    )


def _git_state(repo_root: Path) -> tuple[str | None, bool]:
    """取被测仓库的 HEAD commit 与 dirty 状态（溯源 + lock 判断）。取不到则 (None, False)。

    dirty=True 表示工作区有未提交改动——该 run 的分数复现性存疑，RunRecord 据此标注。
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return (commit or None, bool(status))
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        return None, False


def _resolve_dataset_and_repo(args: argparse.Namespace) -> tuple[Dataset, Path]:
    """从 args 解析数据集 + 被测仓库本地路径（run/sweep 共用）。"""
    dataset = find_dataset(args.repo)
    repo_root = resolve_repo_root(dataset, explicit=args.repo_root)
    return dataset, repo_root


def _build_context(
    args: argparse.Namespace, dataset: Dataset, repo_root: Path
) -> SweepContext:
    """组装 run/sweep 的恒定溯源上下文（profile 提供 L2 元数据，可选）。"""
    commit, dirty = _git_state(repo_root)
    if args.profile:
        profile = _resolve_profile(args.profile)
        embed_model = profile.embedding.model
        dimensions = profile.embedding.dimensions
        embed_endpoint = profile.embedding.endpoint
        db_dialect = profile.backend.db_dialect
        milvus_mode = profile.backend.milvus_mode
        hnsw_m = None  # L1 建库参数不在 profile 的 L0 段；Commit 9 提配置后补
        hnsw_ef = None
        profile_name = profile.name
    else:
        embed_model = dimensions = embed_endpoint = db_dialect = milvus_mode = ""
        dimensions = 0
        hnsw_m = hnsw_ef = None
        profile_name = "(none)"
    return SweepContext(
        repo_name=dataset.name,
        repo_root=repo_root,
        repo_commit=commit,
        repo_dirty=dirty,
        queries_path=dataset.queries_path,
        profile_name=profile_name,
        tag=args.tag,
        embed_model=embed_model,
        embed_dimensions=dimensions,
        embed_endpoint=embed_endpoint,
        db_dialect=db_dialect,
        milvus_mode=milvus_mode,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef,
    )


def _log(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def _cmd_list(args: argparse.Namespace) -> None:
    """列出可用数据集与 profile（只读，不连服务）。"""
    datasets = discover_datasets()
    print("Datasets:")
    if datasets:
        for ds in datasets:
            commit = ds.repository.commit or "?"
            print(
                f"  {ds.alias}  ({ds.questions} q, repo={ds.name}, "
                f"pinned={commit[:10]} {ds.repository.describe})"
            )
    else:
        print("  (none found; datasets land in src/oce/bench/datasets/ — see Commit 8)")

    print("\nProfiles:")
    profiles_dir = default_profiles_dir()
    tomls = sorted(profiles_dir.glob("*.toml")) if profiles_dir.is_dir() else []
    if tomls:
        for path in tomls:
            print(f"  {path.stem}  ({path})")
    else:
        print(f"  (none in {profiles_dir})")


# ---------------------------------------------------------------------------
# serve / reset（服务侧：灌 env + import app）
# ---------------------------------------------------------------------------


def _cmd_serve(args: argparse.Namespace) -> None:
    """起一个隔离的评测服务（阻塞）。注入 OCE_BENCH_HOT_CONFIG=allow 放行热改。"""
    from oce.bench.service import ServiceError, serve

    profile = _resolve_profile(args.profile)
    data_dir = Path(args.data_dir).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        serve(
            profile,
            args.tag,
            data_dir=data_dir,
            port=args.port,
            host=args.host,
            env_file=args.env_file,
        )
    except ServiceError as exc:
        raise BenchCLIError(str(exc)) from exc


def _cmd_reset(args: argparse.Namespace) -> None:
    """清空评测态（DB 表 / 本地文件 + drop collection），带硬闸。"""
    from oce.bench.service import ServiceError, execute_reset, plan_reset

    profile = _resolve_profile(args.profile)
    data_dir = Path(args.data_dir).expanduser().resolve()
    try:
        plan = plan_reset(profile, args.tag, data_dir=data_dir)
    except ServiceError as exc:
        raise BenchCLIError(str(exc)) from exc

    if plan.protected_skipped:
        print(f"[reset] skipping protected collections: {plan.protected_skipped}")
    summary = execute_reset(plan, keep_db=args.keep_db)
    print(f"[reset] done: {summary}")


# ---------------------------------------------------------------------------
# reconfigure（裸热改，调试用）
# ---------------------------------------------------------------------------


def _cmd_reconfigure(args: argparse.Namespace) -> None:
    """对运行中的服务下发一组 L0 patch（read-after-write 校验），打印生效快照。"""
    api_key = _resolve_api_key(args.api_key)

    async def _run() -> None:
        async with BenchClient(args.base_url, api_key, log=_log) as client:
            if args.show:
                snap = await client.get_config()
                print(f"generation={snap.generation}")
                print(json.dumps(snap.effective, ensure_ascii=False, indent=2))
                return
            param_set = param_set_from_args(args.param or [], name="cli")
            if param_set.is_empty():
                raise BenchCLIError("nothing to change; pass --param KEY=VALUE or --show")
            groups = param_set.patch_groups()
            ack = await client.reconfigure(
                retrieval=groups.get("retrieval"),
                flags=groups.get("flags"),
                milvus=groups.get("milvus"),
                rerank=groups.get("rerank"),
            )
            print(f"OK generation={ack.generation} reranker_reloaded={ack.reranker_reloaded}")
            print(json.dumps(ack.effective, ensure_ascii=False, indent=2))

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# run / sweep（客户端侧：连已运行的服务）
# ---------------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> None:
    """单组参数评测：索引（可复用）-> [热改 --param] -> 跑查询 -> 落 RunRecord(JSON+MD)。"""
    dataset, repo_root = _resolve_dataset_and_repo(args)
    context = _build_context(args, dataset, repo_root)
    api_key = _resolve_api_key(args.api_key)
    param_sets: Sequence[ParamSet] = [param_set_from_args(args.param or [], name="run")]
    _run_sweep_blocking(args, context, api_key, param_sets, reuse_index=args.reuse_index)


def _cmd_sweep(args: argparse.Namespace) -> None:
    """多组参数扫描：索引一次 -> for each param set 热改+评分+落记录。秒级切换不重嵌入。"""
    dataset, repo_root = _resolve_dataset_and_repo(args)
    context = _build_context(args, dataset, repo_root)
    api_key = _resolve_api_key(args.api_key)
    try:
        param_sets = load_matrix(args.matrix)
    except SweepError as exc:
        raise BenchCLIError(str(exc)) from exc
    _run_sweep_blocking(args, context, api_key, param_sets, reuse_index=args.reuse_index)


def _run_sweep_blocking(
    args: argparse.Namespace,
    context: SweepContext,
    api_key: str,
    param_sets: Sequence[ParamSet],
    *,
    reuse_index: bool,
) -> None:
    """run/sweep 共用的异步执行外壳：建 client -> run_sweep -> 报告产物路径。"""
    runs_dir = Path(args.runs_dir).expanduser()

    async def _run() -> list[Any]:
        meter = build_meter()
        async with BenchClient(
            args.base_url,
            api_key,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
            log=_log,
        ) as client:
            return await run_sweep(
                client,
                context,
                param_sets=param_sets,
                runs_dir=runs_dir,
                reuse_index=reuse_index,
                embedding_timeout=args.embedding_timeout,
                concurrency=args.concurrency,
                meter=meter,
                log=_log,
            )

    records = asyncio.run(_run())
    print(f"\nwrote {len(records)} run record(s) to {runs_dir.resolve()}")
    for record in records:
        print(f"  {record.run_id}  score={record.score_pct:.1f}%")


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _cmd_compare(args: argparse.Namespace) -> None:
    """读 N 份 RunRecord JSON 出对比矩阵（自动标注差异参数 + 逐题 delta + 可选曲线）。"""
    runs_dir = Path(args.runs).expanduser()
    try:
        records = load_runs(runs_dir, run_ids=args.run_id or None)
        baseline = find_baseline(records, args.baseline)
    except CompareError as exc:
        raise BenchCLIError(str(exc)) from exc

    try:
        markdown = render_compare(records, baseline, param_field=args.param)
    except CompareError as exc:
        raise BenchCLIError(str(exc)) from exc

    if args.output:
        out = Path(args.output).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"wrote comparison to {out}")
    else:
        print(markdown)


# ---------------------------------------------------------------------------
# 解析器构建
# ---------------------------------------------------------------------------


def _add_common_run_args(parser: argparse.ArgumentParser) -> None:
    """run/sweep 共用参数：连服务 + 数据集 + 产物 + 调参 + 资源。"""
    parser.add_argument("--base-url", required=True, help="运行中的评测服务基址")
    parser.add_argument("--api-key", default=None, help="服务 API key（默认取环境变量）")
    parser.add_argument("--repo", required=True, help="数据集别名 / 仓库短名 / 唯一前缀")
    parser.add_argument("--repo-root", default=None, help="覆盖被测仓库本地路径")
    parser.add_argument(
        "--profile", default=None,
        help="记录 L2 元数据（模型/维度/后端）用；不连该 profile 的服务",
    )
    parser.add_argument("--tag", default="run", help="run_id / collection 命名标签")
    parser.add_argument(
        "--runs-dir", default=str(_DEFAULT_RUNS_DIR), help="RunRecord 产物目录",
    )
    parser.add_argument(
        "--reuse-index", action="store_true",
        help="复用服务端既有索引（不重新上传/嵌入），靠 /find-missing 校验",
    )
    parser.add_argument("--concurrency", type=int, default=1, help="查询并发（默认串行）")
    parser.add_argument(
        "--embedding-timeout", type=float, default=3600.0, help="等嵌入完成超时（秒）",
    )
    parser.add_argument("--timeout", type=float, default=120.0, help="单请求超时（秒）")
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="嵌入轮询间隔（秒）",
    )


def build_bench_commands(bench_sub: argparse._SubParsersAction) -> None:
    """在给定的 subparsers 上挂全部 bench 叶子命令（list/serve/reset/.../compare）。

    每个 handler 都经 ``_guard`` 包裹：用户级错误统一打印 + ``SystemExit(2)``，故无论经由
    主 ``oce bench``、``run_bench``，还是测试直接调用，错误处理一致。
    """
    # --- list ---
    p_list = bench_sub.add_parser("list", help="List datasets and profiles")
    p_list.set_defaults(handler=_guard(_cmd_list))

    # --- serve ---
    p_serve = bench_sub.add_parser("serve", help="Start an isolated bench service")
    p_serve.add_argument("--profile", required=True)
    p_serve.add_argument("--tag", default="bench")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--data-dir", default=str(_DEFAULT_BENCH_DATA_DIR))
    p_serve.add_argument("--env-file", default=None)
    p_serve.set_defaults(handler=_guard(_cmd_serve))

    # --- reset ---
    p_reset = bench_sub.add_parser("reset", help="Reset isolated bench state (guarded)")
    p_reset.add_argument("--profile", required=True)
    p_reset.add_argument("--tag", default="bench")
    p_reset.add_argument("--data-dir", default=str(_DEFAULT_BENCH_DATA_DIR))
    p_reset.add_argument(
        "--keep-db", action="store_true",
        help="只 drop collection / 删向量文件，保留元数据 DB",
    )
    p_reset.set_defaults(handler=_guard(_cmd_reset))

    # --- reconfigure ---
    p_reconf = bench_sub.add_parser(
        "reconfigure", help="Hot-patch L0 retrieval params on a running service",
    )
    p_reconf.add_argument("--base-url", required=True)
    p_reconf.add_argument("--api-key", default=None)
    p_reconf.add_argument(
        "--param", action="append", metavar="KEY=VALUE",
        help="可重复；KEY 按 L0 白名单路由到对应组，拼错报错",
    )
    p_reconf.add_argument(
        "--show", action="store_true", help="只打印当前生效配置 + generation，不下发",
    )
    p_reconf.set_defaults(handler=_guard(_cmd_reconfigure))

    # --- run ---
    p_run = bench_sub.add_parser("run", help="Run one evaluation against a service")
    _add_common_run_args(p_run)
    p_run.add_argument(
        "--param", action="append", metavar="KEY=VALUE",
        help="可重复；本次 run 前热改的 L0 参数（空则用服务当前配置）",
    )
    p_run.set_defaults(handler=_guard(_cmd_run))

    # --- sweep ---
    p_sweep = bench_sub.add_parser(
        "sweep", help="Index once, sweep N L0 param sets (matrix TOML)",
    )
    _add_common_run_args(p_sweep)
    p_sweep.add_argument("--matrix", required=True, help="声明式矩阵 TOML（[[set]]）")
    p_sweep.set_defaults(handler=_guard(_cmd_sweep))

    # --- compare ---
    p_cmp = bench_sub.add_parser("compare", help="Compare run records into a matrix")
    p_cmp.add_argument("--runs", default=str(_DEFAULT_RUNS_DIR), help="RunRecord 目录")
    p_cmp.add_argument("--run-id", action="append", help="只比这些 run_id（可重复）")
    p_cmp.add_argument("--baseline", default=None, help="基线 run_id（默认最早一份）")
    p_cmp.add_argument(
        "--param", default=None, metavar="FIELD",
        help="按该字段值排序出参数曲线，如 retrieval.default_top_k",
    )
    p_cmp.add_argument("--output", default=None, help="写 markdown 到此文件（默认打印）")
    p_cmp.set_defaults(handler=_guard(_cmd_compare))


def build_bench_parser(subparsers: argparse._SubParsersAction) -> None:
    """在 oce 主解析器的 subparsers 上挂 ``bench`` 子命令树（``oce bench <cmd>``）。"""
    bench = subparsers.add_parser("bench", help="Retrieval evaluation harness")
    bench_sub = bench.add_subparsers(dest="bench_command", required=True)
    build_bench_commands(bench_sub)


def run_bench(argv: Sequence[str]) -> int:
    """``oce bench ...`` 的独立入口（便于测试与脚本直接调用）。返回退出码。

    叶子命令直接挂在顶层（``run_bench(["serve", ...])``，无需重复 ``bench``）。handler 已经
    过 ``_guard``：用户级错误会 ``SystemExit(2)``，在此转成返回值 2，不抛栈。
    """
    parser = argparse.ArgumentParser(prog="oce bench", add_help=True)
    sub = parser.add_subparsers(dest="bench_command", required=True)
    build_bench_commands(sub)
    args = parser.parse_args(list(argv))
    try:
        args.handler(args)
        return 0
    except SystemExit as exc:  # _guard 抛的退出码
        return int(exc.code or 0)
