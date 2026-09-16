"""service.py 纯规划层测试：reset 硬闸 + ResetPlan 构造（不碰 DB / Milvus / uvicorn）。

只测**纯函数**（assert_resettable_db_url / resettable_collections / plan_reset /
prepare_serve_env），脏执行（execute_reset / serve）依赖真后端，留给集成测试或手验。

核心安全断言（照搬 reset_bench.py 的防呆红线）：
1. DB URL 不含 ``oce_bench`` -> assert_resettable_db_url 拒绝（绝不碰生产库）。
2. PROTECTED collection 永不进 plan.collections（即使误传也只跳过）。
3. sqlite/milvus-lite 模式 -> plan 给本地文件路径；postgres/server 模式 -> 给 endpoint。
4. plan_reset 在**规划期**就过硬闸（DB URL 闸），不等到 execute 才报错。
5. prepare_serve_env 注入 OCE_BENCH_HOT_CONFIG=allow（放行热改）+ LOG_LEVEL=INFO。
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from oce.bench.profiles import (
    Backend,
    Embedding,
    Isolation,
    LLM,
    Pipeline,
    Profile,
    Rerank,
    Service,
    _SecretRef,
)
from oce.bench.service import (
    PROTECTED_COLLECTIONS,
    RESET_TABLES,
    ServiceError,
    assert_resettable_db_url,
    plan_reset,
    prepare_serve_env,
    resettable_collections,
)


@pytest.fixture(autouse=True)
def _restore_environ():
    """prepare_serve_env 直接 os.environ.update（非 monkeypatch）-> 快照还原防跨测污染。"""
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)


# ---------------------------------------------------------------------------
# 硬闸①：DB URL 必须含 oce_bench
# ---------------------------------------------------------------------------


def test_assert_resettable_db_url_accepts_bench():
    assert_resettable_db_url("postgresql+asyncpg://u:p@h/oce_bench")  # no raise
    assert_resettable_db_url("sqlite+aiosqlite:////data/oce_bench.db")  # no raise


def test_assert_resettable_db_url_rejects_production():
    with pytest.raises(ServiceError, match="database name must be"):
        assert_resettable_db_url("postgresql+asyncpg://u:p@h/oce_production")


def test_assert_resettable_db_url_rejects_bench_marker_in_password():
    with pytest.raises(ServiceError, match="database name must be"):
        assert_resettable_db_url(
            "postgresql+asyncpg://u:oce_bench_secret@h/oce_production"
        )


def test_assert_resettable_db_url_rejects_empty():
    with pytest.raises(ServiceError):
        assert_resettable_db_url("")


# ---------------------------------------------------------------------------
# 硬闸②：PROTECTED collection 永不 drop
# ---------------------------------------------------------------------------


def test_resettable_collections_splits_protected():
    droppable, protected = resettable_collections(
        ["bench_local_chunks", "oce_chunks", "bench_local_paths", "oce_paths_v1"]
    )
    assert droppable == ["bench_local_chunks", "bench_local_paths"]
    assert protected == ["oce_chunks", "oce_paths_v1"]


def test_resettable_collections_all_safe():
    droppable, protected = resettable_collections(["bench_x_chunks"])
    assert droppable == ["bench_x_chunks"]
    assert protected == []


def test_resettable_collections_empty():
    assert resettable_collections([]) == ([], [])


def test_protected_collections_are_real_production_names():
    # 防回归：PROTECTED 集合必须含真实生产 collection（reset 绝不能 drop 它们）
    assert "oce_chunks" in PROTECTED_COLLECTIONS
    assert "oce_paths_v1" in PROTECTED_COLLECTIONS


def test_reset_tables_covers_content_and_metrics():
    # 防回归：reset 必须清内容表（强制重索引）+ metrics 表（成本核算干净）
    assert "blob_staging" in RESET_TABLES
    assert "chunks" in RESET_TABLES
    assert "token_usage_metrics" in RESET_TABLES
    assert "resource_samples" in RESET_TABLES


# ---------------------------------------------------------------------------
# plan_reset：sqlite/lite 本地模式
# ---------------------------------------------------------------------------


def _local_profile(tmp_path: Path) -> Profile:
    return Profile(
        name="local",
        backend=Backend(
            db_dialect="sqlite+aiosqlite",
            db_path="{data_dir}/oce_bench.db",
            milvus_mode="lite",
            milvus_path="{data_dir}/oce_bench_milvus.db",
        ),
        isolation=Isolation(db_name=None, redis_db=0, collection_prefix="bench_local"),
        service=Service(host="127.0.0.1", port=8987, worker_enabled=False),
        embedding=Embedding(model="f2llm-v2-0.6b", dimensions=1024),
        rerank=Rerank(), llm=LLM(), pipeline=Pipeline(),
        source_path=tmp_path / "local.toml",
    )


def test_plan_reset_sqlite_gives_local_file_paths(tmp_path: Path):
    profile = _local_profile(tmp_path)
    data_dir = tmp_path / "data"
    plan = plan_reset(profile, "v1", data_dir=data_dir)
    assert plan.is_sqlite is True
    assert plan.sqlite_db_path is not None
    assert plan.sqlite_db_path.name == "oce_bench.db"
    assert plan.milvus_lite_path is not None
    assert plan.milvus_lite_path.name == "oce_bench_milvus.db"
    # collection 名带 tag 隔离
    assert plan.collections == ("bench_local_v1_chunks", "bench_local_v1_paths")
    assert plan.protected_skipped == ()


def test_plan_reset_sqlite_db_url_contains_bench(tmp_path: Path):
    profile = _local_profile(tmp_path)
    plan = plan_reset(profile, "v1", data_dir=tmp_path / "data")
    assert "oce_bench" in plan.db_url
    assert plan.db_url.startswith("sqlite")


def test_plan_reset_runs_db_url_guard_at_plan_time(tmp_path: Path):
    """DB URL 闸在规划期触发（不等 execute）：profile 指向非 bench 库 -> plan_reset 即拒绝。"""
    profile = Profile(
        name="bad",
        backend=Backend(
            db_dialect="sqlite+aiosqlite",
            db_path="{data_dir}/production.db",  # 不含 oce_bench
            milvus_mode="lite",
            milvus_path="{data_dir}/m.db",
        ),
        isolation=Isolation(collection_prefix="bench"),
        service=Service(), embedding=Embedding(), rerank=Rerank(), llm=LLM(),
        pipeline=Pipeline(), source_path=tmp_path / "bad.toml",
    )
    with pytest.raises(ServiceError, match="database name must be"):
        plan_reset(profile, "v1", data_dir=tmp_path / "data")


def test_plan_reset_rejects_sqlite_path_outside_data_dir(tmp_path: Path):
    profile = _local_profile(tmp_path)
    profile = Profile(
        **{
            **profile.__dict__,
            "backend": Backend(
                db_dialect="sqlite+aiosqlite",
                db_path=str(tmp_path / "outside" / "oce_bench.db"),
                milvus_mode="lite",
                milvus_path="{data_dir}/oce_bench_milvus.db",
            ),
        }
    )
    with pytest.raises(ServiceError, match="inside data_dir"):
        plan_reset(profile, "v1", data_dir=tmp_path / "data")


def test_plan_reset_rejects_milvus_lite_path_outside_data_dir(tmp_path: Path):
    profile = _local_profile(tmp_path)
    profile = Profile(
        **{
            **profile.__dict__,
            "backend": Backend(
                db_dialect="sqlite+aiosqlite",
                db_path="{data_dir}/oce_bench.db",
                milvus_mode="lite",
                milvus_path=str(tmp_path / "outside" / "milvus.db"),
            ),
        }
    )
    with pytest.raises(ServiceError, match="inside data_dir"):
        plan_reset(profile, "v1", data_dir=tmp_path / "data")


def test_plan_reset_protected_collection_skipped(tmp_path: Path):
    """collection 前缀若与生产名撞车，受保护项被剔除并记入 protected_skipped。"""
    profile = Profile(
        name="clash",
        backend=Backend(
            db_dialect="sqlite+aiosqlite",
            db_path="{data_dir}/oce_bench.db",
            milvus_mode="lite",
            milvus_path="{data_dir}/m.db",
        ),
        # prefix+tag 拼出正好是生产名 oce_chunks / oce_paths_v1 的几乎不可能，
        # 故直接用 isolation 让候选集合落到受保护名上
        isolation=Isolation(collection_prefix="oce", queue_template="q"),
        service=Service(), embedding=Embedding(), rerank=Rerank(), llm=LLM(),
        pipeline=Pipeline(), source_path=tmp_path / "clash.toml",
    )
    # tag="chunks" -> 候选 ("oce_chunks_chunks","oce_chunks_paths")，均非受保护 -> 全可 drop
    plan = plan_reset(profile, "chunks", data_dir=tmp_path / "data")
    assert plan.protected_skipped == ()
    assert plan.collections == ("oce_chunks_chunks", "oce_chunks_paths")


# ---------------------------------------------------------------------------
# plan_reset：postgres/server 容器模式
# ---------------------------------------------------------------------------


def test_plan_reset_postgres_gives_endpoint(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DB_PW", "secret")
    profile = Profile(
        name="docker",
        backend=Backend(
            db_dialect="postgresql+asyncpg",
            db_host="localhost", db_port=25432, db_user="oce",
            db_password=_SecretRef(env_var="DB_PW"),
            milvus_mode="server", milvus_endpoint="http://localhost:19530",
            redis_host="localhost", redis_port=26379,
        ),
        isolation=Isolation(db_name="oce_bench", redis_db=15, collection_prefix="bench_dk"),
        service=Service(worker_enabled=True),
        embedding=Embedding(dimensions=1024),
        rerank=Rerank(), llm=LLM(), pipeline=Pipeline(),
        source_path=tmp_path / "docker.toml",
    )
    plan = plan_reset(profile, "sw", data_dir=tmp_path / "data")
    assert plan.is_sqlite is False
    assert plan.milvus_endpoint == "http://localhost:19530"
    assert plan.sqlite_db_path is None  # 容器模式不给本地文件
    assert plan.collections == ("bench_dk_sw_chunks", "bench_dk_sw_paths")
    assert "oce_bench" in plan.db_url


# ---------------------------------------------------------------------------
# prepare_serve_env：热改闸 + 日志级别
# ---------------------------------------------------------------------------


def test_prepare_serve_env_sets_hot_config_gate(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OCE_BENCH_HOT_CONFIG", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    profile = _local_profile(tmp_path)
    env = prepare_serve_env(profile, "v1", data_dir=tmp_path / "data")
    # 热改闸放行（正常 oce serve 永不设此变量 -> 生产不可能被热改检索行为）
    assert os.environ["OCE_BENCH_HOT_CONFIG"] == "allow"
    assert os.environ["LOG_LEVEL"] == "INFO"
    # env 里含 bench 基础设施键（DB_URL / collection）
    assert "DB_URL" in env
    assert env["MILVUS_COLLECTION_NAME"] == "bench_local_v1_chunks"
    # 维度一致性
    assert env["EMBED_DIMENSIONS"] == env["MILVUS_DENSE_DIM"]


def test_prepare_serve_env_hot_config_disabled(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OCE_BENCH_HOT_CONFIG", raising=False)
    profile = _local_profile(tmp_path)
    prepare_serve_env(profile, "v1", data_dir=tmp_path / "data", hot_config=False)
    # hot_config=False -> 不放行热改
    assert os.environ.get("OCE_BENCH_HOT_CONFIG") != "allow"


def test_prepare_serve_env_preserves_existing_log_level(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    profile = _local_profile(tmp_path)
    prepare_serve_env(profile, "v1", data_dir=tmp_path / "data")
    # setdefault：用户已设 DEBUG 则尊重，不覆盖成 INFO
    assert os.environ["LOG_LEVEL"] == "DEBUG"
