"""评测服务生命周期：serve（起隔离服务）与 reset（清评测态，带硬闸）。

取代 scratch/embed_bench/{serve_bench,reset_bench}.py，但**数据驱动、无明文、无绝对路径**：
配置来自 profile（Commit 5），环境注入走 apply_profile（import app 之前），reset 的破坏性
动作前置于纯函数硬闸。

设计原则——**纯规划 / 脏执行分离**：
- ``prepare_serve_env`` / ``plan_reset`` / ``assert_resettable_db_url`` / ``resettable_collections``
  是纯函数，返回结构化计划，单元测试可全覆盖（不碰 DB / Milvus / uvicorn）。
- ``serve`` / ``execute_reset`` 是脏执行（uvicorn.run / sqlalchemy / pymilvus），重依赖惰性
  导入，集成测试或真命令行才触发。

reset 硬闸（照搬 reset_bench.py，是防呆红线）：
1. 拒绝任何 DB URL 不含 ``oce_bench`` 的目标（绝不碰生产库）。
2. 拒绝 drop PROTECTED 集合（oce_chunks* / oce_paths_v1 等真实生产 collection）。
3. sqlite/milvus-lite 本地文件模式：只删 data_dir 下带 bench 名的文件，且必须存在才删。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from oce.bench.profiles import Profile, apply_profile

# 生产 collection 红线：reset 绝不 drop 这些（照搬 reset_bench.py 的 PROTECTED）。
PROTECTED_COLLECTIONS: frozenset[str] = frozenset({
    "oce_chunks",
    "oce_chunks_qwen3",
    "oce_chunks_qwen3_hybrid",
    "oce_paths_v1",
    "oce_openclaw_eval_20260812b",
})

# reset 要清空的内容/状态表（照搬 reset_bench.py 的 TABLES）。blob 状态是全局的（非按
# collection），必须清才能强制重新索引。metrics 表一并清，保证成本核算干净。
RESET_TABLES: tuple[str, ...] = (
    "blob_staging",
    "blob_chunks",
    "chunks",
    "blobs",
    "chain_members",
    "chains",
    "symbol_occurrences",
    "token_usage_metrics",
    "retrieval_metrics",
    "api_call_metrics",
    "resource_samples",
)


class ServiceError(Exception):
    """serve/reset 失败：硬闸拦截、依赖缺失、生命周期错误。消息面向用户。"""


# ---------------------------------------------------------------------------
# serve：纯环境准备 + 脏执行
# ---------------------------------------------------------------------------


def prepare_serve_env(
    profile: Profile,
    tag: str,
    *,
    data_dir: Path,
    env_file: str | None = None,
    hot_config: bool = True,
) -> dict[str, str]:
    """在 import app 之前准备好全部 os.environ（纯规划：返回注入的键，同时写 os.environ）。

    在 profiles.apply_profile 之上，追加评测服务专属的闸与隔离：
    - ``OCE_BENCH_HOT_CONFIG=allow``：放行 /admin/bench/retrieval-config 热改（hot_config
      为真时）。正常 ``oce serve`` 永不设此变量，故生产部署不可能被热改检索行为。
    - ``LOG_LEVEL``：评测服务默认 INFO（serve_bench 行为）。

    必须在 import oce.main / 读取 settings 之前调用（pydantic-settings 进程 env 优先，且
    get_settings/get_container 带 lru_cache 无 cache_clear）。
    """
    env = apply_profile(profile, tag, data_dir=data_dir, env_file=env_file)
    if hot_config:
        os.environ["OCE_BENCH_HOT_CONFIG"] = "allow"
    os.environ.setdefault("LOG_LEVEL", "INFO")
    return env


def serve(
    profile: Profile,
    tag: str,
    *,
    data_dir: Path,
    port: int | None = None,
    host: str | None = None,
    env_file: str | None = None,
) -> None:
    """起一个隔离的评测服务（阻塞，Ctrl-C 停）。

    顺序铁律：prepare_serve_env（灌 env）→ run_migrations → **之后**才 import uvicorn/app，
    否则 settings/engine 会读到未注入的默认配置（serve_bench 的核心手法，此处数据驱动版）。
    """
    env = prepare_serve_env(profile, tag, data_dir=data_dir, env_file=env_file)
    effective_host = host or profile.service.host
    effective_port = port or profile.service.port

    db_url = env["DB_URL"]
    print(
        f"[bench serve] tag={tag} port={effective_port} "
        f"model={profile.embedding.model} dim={profile.embedding.dimensions} "
        f"collection={env['MILVUS_COLLECTION_NAME']}",
        flush=True,
    )

    # sqlite 文件可直接幂等迁移；postgres 也走同一入口（reset 后是空库）
    from oce.infrastructure.persistence.migrations import run_migrations

    run_migrations(db_url)

    # env 就绪后才 import app（关键顺序）
    import uvicorn

    uvicorn.run(
        "oce.main:app",
        host=effective_host,
        port=effective_port,
        reload=False,
        log_level="info",
        access_log=False,
    )


# ---------------------------------------------------------------------------
# reset：纯规划（含硬闸）+ 脏执行
# ---------------------------------------------------------------------------


def assert_resettable_db_url(db_url: str) -> None:
    """硬闸①：DB URL 必须含 ``oce_bench``，否则拒绝（绝不碰生产库）。"""
    if "oce_bench" not in db_url:
        raise ServiceError(
            f"refusing to reset: DB_URL does not contain 'oce_bench': {db_url}"
        )


def resettable_collections(names: Sequence[str]) -> tuple[list[str], list[str]]:
    """硬闸②：把待 drop 集合分成 ``(可安全 drop, 受保护跳过)``。

    受保护的生产 collection 永不 drop —— 即使调用方误传也只跳过并告警。
    """
    droppable: list[str] = []
    protected: list[str] = []
    for name in names:
        if name in PROTECTED_COLLECTIONS:
            protected.append(name)
        else:
            droppable.append(name)
    return droppable, protected


@dataclass(frozen=True)
class ResetPlan:
    """一次 reset 的完整计划（纯数据，执行前已过硬闸，自包含不依赖 ambient env）。

    本地模式（sqlite + milvus-lite）与容器模式（postgres + milvus-server）用互斥字段表达，
    执行期按 ``is_sqlite`` 分流，不做脆弱的文件名嗅探。
    """

    db_url: str
    is_sqlite: bool
    # 待 drop 的 collection 名（已过受保护闸）
    collections: tuple[str, ...]
    protected_skipped: tuple[str, ...]
    # 容器模式：milvus server 端点（drop collection 用）
    milvus_endpoint: str = ""
    # 本地模式：sqlite 库文件 / milvus-lite 文件（存在才删）
    sqlite_db_path: Path | None = None
    milvus_lite_path: Path | None = None


def plan_reset(profile: Profile, tag: str, *, data_dir: Path) -> ResetPlan:
    """从 profile + tag 推出 reset 计划，**在规划期就过硬闸**（不等到执行才报错）。

    集合名 = ``{prefix}_{tag}_{chunks,paths}``（与 build_env 同源，保证删的正是评测建的）。
    DB_URL 经 assert_resettable_db_url；集合经 resettable_collections 剔除受保护项。
    """
    from oce.bench.profiles import build_env

    env = build_env(profile, tag, data_dir=data_dir)
    db_url = env["DB_URL"]
    assert_resettable_db_url(db_url)

    candidates = (
        env["MILVUS_COLLECTION_NAME"],
        env["MILVUS_PATH_COLLECTION_NAME"],
    )
    droppable, protected = resettable_collections(candidates)

    is_sqlite = db_url.startswith("sqlite")
    milvus_endpoint = env["MILVUS_ENDPOINT"]
    milvus_is_lite = not milvus_endpoint.startswith("http")

    if is_sqlite:
        # sqlite:/// + 绝对路径 -> 去掉 '///' 前缀取文件路径
        sqlite_db_path = Path(db_url.split("///", 1)[-1])
        milvus_lite_path = Path(milvus_endpoint) if milvus_is_lite else None
        return ResetPlan(
            db_url=db_url,
            is_sqlite=True,
            collections=tuple(droppable),
            protected_skipped=tuple(protected),
            sqlite_db_path=sqlite_db_path,
            milvus_lite_path=milvus_lite_path,
        )

    return ResetPlan(
        db_url=db_url,
        is_sqlite=False,
        collections=tuple(droppable),
        protected_skipped=tuple(protected),
        milvus_endpoint=milvus_endpoint,
    )


def execute_reset(plan: ResetPlan, *, keep_db: bool = False) -> dict[str, object]:
    """执行 reset 计划（脏：sqlalchemy TRUNCATE / 删本地文件 + pymilvus drop）。

    ``keep_db=True``：只 drop collection / 删向量文件，不清元数据 DB（保留 blob 记录，
    用于只重建向量索引）。重依赖惰性导入——单元测试只验 plan_reset 的纯规划，不触发此函数。

    返回动作摘要（截断表数 / 删的文件 / drop 的 collection / 跳过的受保护项），供 CLI 打印。
    """
    deleted: list[str] = []
    truncated = 0
    dropped: list[str] = []

    if plan.is_sqlite:
        # 本地模式：删文件（存在才删）。DB 文件受 keep_db 保护；向量文件总是删。
        if not keep_db and plan.sqlite_db_path is not None:
            if plan.sqlite_db_path.exists():
                plan.sqlite_db_path.unlink()
                deleted.append(str(plan.sqlite_db_path))
        if plan.milvus_lite_path is not None and plan.milvus_lite_path.exists():
            plan.milvus_lite_path.unlink()
            deleted.append(str(plan.milvus_lite_path))
    else:
        # 容器模式：TRUNCATE 表 + drop collection。
        if not keep_db:
            truncated = _truncate_postgres(plan.db_url)
        dropped = _drop_milvus_collections(
            list(plan.collections), plan.milvus_endpoint
        )

    return {
        "truncated_tables": truncated,
        "deleted_files": deleted,
        "dropped_collections": dropped,
        "protected_skipped": list(plan.protected_skipped),
    }


def _truncate_postgres(db_url: str) -> int:
    """TRUNCATE 内容/状态表（RESTART IDENTITY CASCADE）；返回实际清空的表数。"""
    import asyncio

    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    async def _run() -> int:
        engine = create_async_engine(db_url)
        try:
            async with engine.begin() as conn:
                existing = set(
                    (
                        await conn.execute(
                            text(
                                "SELECT table_name FROM information_schema.tables "
                                "WHERE table_schema='public'"
                            )
                        )
                    ).scalars().all()
                )
                targets = [t for t in RESET_TABLES if t in existing]
                if targets:
                    await conn.execute(
                        text("TRUNCATE " + ", ".join(targets) + " RESTART IDENTITY CASCADE")
                    )
                return len(targets)
        finally:
            await engine.dispose()

    return asyncio.run(_run())


def _drop_milvus_collections(names: list[str], endpoint: str) -> list[str]:
    """drop 存在的 collection（受保护项已在 plan 期剔除）；返回实际 drop 的名字。"""
    from pymilvus import MilvusClient

    if not endpoint:
        raise ServiceError("milvus endpoint missing from reset plan")
    client = MilvusClient(uri=endpoint)
    existing = set(client.list_collections())
    dropped: list[str] = []
    for name in names:
        if name in existing:
            client.drop_collection(name)
            dropped.append(name)
    return dropped
