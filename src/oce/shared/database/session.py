"""SQLAlchemy 引擎、会话工厂和 ORM 基类。"""

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from oce.shared.config import get_settings
from oce.shared.database.postgresql_adapter import PostgreSQLAdapter
from oce.shared.database.sqlite_adapter import SQLiteAdapter


def _create_engine():
    settings = get_settings().database
    if settings.is_sqlite:
        return SQLiteAdapter().create_engine(settings.url, echo=settings.echo)
    return PostgreSQLAdapter(
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
    ).create_engine(settings.url, echo=settings.echo)


engine = _create_engine()
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
Base = declarative_base()
