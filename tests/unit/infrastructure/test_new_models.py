"""验证新表结构在 SQLite 中是否能创建

测试 infrastructure/models.py 的表定义：
- 无 PG 专有类型（TSVECTOR/HALFVEC/Computed）
- PostgreSQL/SQLite 兼容
"""

import pytest
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_new_models_create_in_sqlite():
    """验证新表结构在 SQLite 中能创建"""
    from oce.infrastructure.persistence.models import (
        BlobModel,
        ChunkModel,
        BlobChunkModel,
        ChainModel,
        ChainMemberModel,
    )
    from sqlalchemy.orm import declarative_base
    
    # 创建临时 Base（避免污染全局 Base.metadata）
    Base = declarative_base()
    
    # 复制表定义
    class TestBlobModel(Base):
        __table__ = BlobModel.__table__.to_metadata(Base.metadata)
    
    class TestChunkModel(Base):
        __table__ = ChunkModel.__table__.to_metadata(Base.metadata)
    
    class TestBlobChunkModel(Base):
        __table__ = BlobChunkModel.__table__.to_metadata(Base.metadata)
    
    class TestChainModel(Base):
        __table__ = ChainModel.__table__.to_metadata(Base.metadata)
    
    class TestChainMemberModel(Base):
        __table__ = ChainMemberModel.__table__.to_metadata(Base.metadata)
    
    # 用 SQLite 内存数据库创建表
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=True)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # 验证表已创建（能执行简单查询）
    from sqlalchemy import text
    async with engine.connect() as conn:
        # 验证 5 张表都存在
        result = await conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = {row[0] for row in result}
        
        assert "blobs" in tables
        assert "chunks" in tables
        assert "blob_chunks" in tables
        assert "chains" in tables
        assert "chain_members" in tables
    
    await engine.dispose()


@pytest.mark.asyncio
async def test_new_models_create_in_postgresql():
    """验证新表结构在 PostgreSQL 中也能创建（如果配置了 PG）"""
    import uuid
    import os
    from dotenv import dotenv_values

    # 直接读 .env 文件，绕过 pytest 的环境变量覆盖
    env_config = dotenv_values(".env")
    db_url = env_config.get("DB_URL")

    if not db_url or "postgresql" not in str(db_url):
        pytest.skip(f"No PostgreSQL configured in .env (got: {db_url})")
    
    from oce.shared.database.session import Base
    from sqlalchemy import text

    # 导入模型让 Base.metadata 包含表定义
    from oce.infrastructure.persistence.models import (
        BlobModel,
        ChunkModel,
        BlobChunkModel,
        ChainModel,
        ChainMemberModel,
    )

    schema = f"oce_model_test_{uuid.uuid4().hex}"
    engine = create_async_engine(db_url, echo=True)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
            await conn.run_sync(
                lambda sync_conn: Base.metadata.create_all(
                    sync_conn.execution_options(
                        schema_translate_map={None: schema},
                    )
                )
            )
        async with engine.connect() as conn:
            tables = set(
                await conn.scalars(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = :schema"
                    ),
                    {"schema": schema},
                )
            )
        assert {"blobs", "chunks", "blob_chunks", "chains", "chain_members"} <= tables
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()
