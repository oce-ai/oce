"""Alembic async migration environment.

LOCATION NOTE: this file lives inside the ``oce`` package (``src/oce/alembic/``)
so that ``uv tool install`` wheels carry the migration scripts; ``oce serve``
runs ``alembic upgrade head`` programmatically without a repository checkout.
"""

from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from oce.shared.database.session import Base
from oce.shared.config import get_settings
import oce.infrastructure.persistence.models  # noqa: F401

config = context.config
config.set_main_option("sqlalchemy.url", get_settings().database.url)
if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata
VERSION_TABLE = "oce_alembic_version"


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        version_table=VERSION_TABLE,
        compare_type=True,
        compare_comments=False,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table=VERSION_TABLE,
        compare_type=True,
        compare_comments=False,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(run_migrations)
    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
