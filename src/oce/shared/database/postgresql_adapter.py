"""PostgreSQL async engine configuration."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


class PostgreSQLAdapter:
    def __init__(self, pool_size: int = 30, max_overflow: int = 30) -> None:
        self.pool_size = pool_size
        self.max_overflow = max_overflow

    def create_engine(self, url: str, **kwargs: Any) -> AsyncEngine:
        options = {
            "echo": kwargs.get("echo", False),
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_pre_ping": True,
            "pool_timeout": 5,
            "pool_recycle": 1800,
        }
        options.update(kwargs)
        return create_async_engine(url, **options)
