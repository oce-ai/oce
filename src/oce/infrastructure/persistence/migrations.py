"""编程式 Alembic 迁移入口（个人模式 ``oce serve`` 自动迁移用）。

服务模式仍由 compose/手动执行 ``uv run alembic upgrade head``；本模块不会自动
修改 PostgreSQL，只对 SQLite 个人库做旧版（create_all 时代）baseline 兼容。
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from loguru import logger
from sqlalchemy import create_engine, inspect

# 迁移脚本随 wheel 一起发布（src/oce/alembic），运行时用包内路径而非仓库 CWD，
# 保证 `uv tool install` 安装的环境也能离线迁移。
_SCRIPT_DIR = Path(__file__).resolve().parents[2] / "alembic"

_VERSION_TABLE = "oce_alembic_version"


def _sync_url(url: str) -> str:
    """aiosqlite 驱动 URL 换成同步驱动，供基线探测使用。"""
    return url.replace("sqlite+aiosqlite://", "sqlite://", 1)


def _needs_baseline(url: str) -> bool:
    """旧 create_all 库没有版本表但有业务表 → 视为等价 head，stamp 而非重放。

    仅适用于 SQLite 个人库：create_all 产出的 schema 与 models 一致，迁移链 head
    也与 models 一致，因此 stamp 安全。PostgreSQL 由服务模式手动维护版本表。
    """
    if not url.startswith("sqlite"):
        return False
    engine = create_engine(_sync_url(url))
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()
    return bool(tables) and _VERSION_TABLE not in tables


def run_migrations(url: str) -> None:
    """把数据库升级到最新 schema；返回前保证版本表存在并指向 head。

    env.py 会从 settings 读取 DB_URL（``oce serve`` 启动前已把个人模式 URL 写入
    环境变量），与传入的 url 指向同一数据库，二者必须一致。
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(_SCRIPT_DIR))

    if _needs_baseline(url):
        logger.info("Legacy SQLite database found; stamping {}", _VERSION_TABLE)
        command.stamp(cfg, "head")
    command.upgrade(cfg, "head")
