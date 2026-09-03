"""SQLite async engine configuration."""

from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool


class SQLiteAdapter:
    def create_engine(self, url: str, **kwargs: Any) -> AsyncEngine:
        """创建 SQLite 异步引擎，针对个人模式优化以避免锁冲突。

        - WAL 模式：允许读写并发
        - busy_timeout：等待锁最多 30 秒
        - StaticPool：单连接池，避免多连接死锁
        """
        engine = create_async_engine(
            url,
            connect_args={
                "check_same_thread": False,
                "timeout": 30.0,  # busy_timeout 30 秒
            },
            poolclass=StaticPool,  # 个人模式单连接，避免死锁
            echo=kwargs.get("echo", False),
        )

        # 启用 WAL 模式以支持并发读写
        @event.listens_for(engine.sync_engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")  # WAL 模式下可以降低同步级别
            cursor.close()

        return engine
