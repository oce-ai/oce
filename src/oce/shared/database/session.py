"""SQLAlchemy 引擎、会话工厂和 ORM 基类。

engine / session 工厂经 ``get_engine()`` / ``get_session_factory()`` **惰性构造**（首次
调用时读 settings 并缓存），绝不在 import 期建：本模块位于 ``oce.bench.cli`` 的传递
import 链上（cli -> client -> blobs -> application.service -> ... -> session），而
``oce bench serve`` 要等 handler 运行后才注入 profile env。import 期读配置会把
get_settings 的 lru_cache 冻结在注入前的值上，整个服务静默跑在后端默认配置
（实测：local profile 起的服务连了 docker Milvus 19530 与仓库 .env 的远端嵌入端点）。
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from oce.shared.config import get_settings
from oce.shared.database.postgresql_adapter import PostgreSQLAdapter
from oce.shared.database.sqlite_adapter import SQLiteAdapter

Base = declarative_base()

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """返回进程级共享 engine；首次调用时按当前 settings 构造。"""
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings().database
        if settings.is_sqlite:
            _engine = SQLiteAdapter().create_engine(settings.url, echo=settings.echo)
        else:
            _engine = PostgreSQLAdapter(
                pool_size=settings.pool_size,
                max_overflow=settings.max_overflow,
            ).create_engine(settings.url, echo=settings.echo)
        _session_factory = async_sessionmaker(
            _engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """返回进程级共享 session 工厂（绑定 get_engine() 同源构造的 engine）。"""
    if _session_factory is None:
        get_engine()
        assert _session_factory is not None
    return _session_factory
