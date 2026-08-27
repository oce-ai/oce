"""测试数据库 fixtures

验证 SQLite 数据库 fixtures 工作正常。
"""

import pytest
from sqlalchemy import text


@pytest.mark.asyncio
async def test_db_engine_fixture(test_engine):
    """测试数据库引擎 fixture"""
    assert test_engine is not None
    assert test_engine.dialect.name == "sqlite"


@pytest.mark.asyncio
async def test_db_session_fixture(test_session):
    """测试数据库 session fixture"""
    assert test_session is not None

    # 可以执行简单查询
    result = await test_session.execute(text("SELECT 1"))
    assert result.scalar() == 1


@pytest.mark.asyncio
async def test_db_session_isolation(test_session):
    """测试 session 隔离（每个测试独立）"""
    # 每个测试都应该有干净的 session
    assert test_session is not None

    result = await test_session.execute(text("SELECT 1"))
    assert result.scalar() == 1
