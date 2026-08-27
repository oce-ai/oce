"""SQLite async engine configuration."""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine


class SQLiteAdapter:
    def create_engine(self, url: str, **kwargs: Any) -> AsyncEngine:
        return create_async_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=kwargs.get("echo", False),
        )
