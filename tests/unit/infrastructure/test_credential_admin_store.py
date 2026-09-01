"""SqlCredentialAdminStore CRUD / duplicate 测试。"""

from __future__ import annotations

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from oce.application.credential_admin import (
    CredentialCreate,
    CredentialDuplicate,
    CredentialUpdate,
)
from oce.infrastructure.persistence.credential_admin_store import (
    SqlCredentialAdminStore,
    _hash_key,
)
from oce.shared.database.session import Base
from oce.shared.errors import CredentialConflictError


async def _store():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    return engine, SqlCredentialAdminStore(
        async_sessionmaker(engine, expire_on_commit=False)
    )


def _create(**overrides) -> CredentialCreate:
    data = dict(
        kind="embed",
        name="primary",
        api_key="sk-secret-1234",
        provider="siliconflow",
        endpoint="https://example.test/v1/embeddings",
        model="embedding-model",
        dimensions=1024,
    )
    data.update(overrides)
    return CredentialCreate(**data)


async def test_create_returns_masked_record():
    engine, store = await _store()
    record = await store.create(_create())
    assert record.id > 0
    assert record.api_key_last4 == "1234"
    assert record.provider == "siliconflow"
    assert record.dimensions == 1024
    await engine.dispose()


async def test_list_orders_by_priority():
    engine, store = await _store()
    await store.create(_create(name="low", api_key="k-low", priority=200))
    await store.create(_create(name="high", api_key="k-high", priority=1))
    records = await store.list()
    assert [r.name for r in records] == ["high", "low"]
    await engine.dispose()


async def test_update_changes_fields_and_rehashes_key():
    engine, store = await _store()
    created = await store.create(_create())
    updated = await store.update(
        created.id, CredentialUpdate(status="disabled", api_key="sk-new-9999")
    )
    assert updated is not None
    assert updated.status == "disabled"
    assert updated.api_key_last4 == "9999"
    await engine.dispose()


async def test_update_missing_returns_none():
    engine, store = await _store()
    assert await store.update(999, CredentialUpdate(status="disabled")) is None
    await engine.dispose()


async def test_delete_removes_row():
    engine, store = await _store()
    created = await store.create(_create())
    assert await store.delete(created.id) is True
    assert await store.list() == []
    assert await store.delete(created.id) is False
    await engine.dispose()


async def test_duplicate_copies_channel_config():
    engine, store = await _store()
    src = await store.create(
        _create(kind="rerank", endpoint="https://r.test", model="rr")
    )
    clone = await store.duplicate(
        src.id, CredentialDuplicate(name="secondary", api_key="sk-clone-5678")
    )
    assert clone is not None
    assert clone.id != src.id
    assert clone.name == "secondary"
    assert clone.api_key_last4 == "5678"
    assert clone.kind == "rerank"
    assert clone.endpoint == "https://r.test"
    assert clone.model == "rr"
    await engine.dispose()


async def test_duplicate_overrides_kind_and_reuses_source_key():
    """复制时覆盖 kind/model、省略 api_key：继承源 key，因 kind 不同不撞唯一约束。"""
    engine, store = await _store()
    src = await store.create(_create(api_key="sk-shared-1234"))
    clone = await store.duplicate(
        src.id,
        CredentialDuplicate(
            name="as-rerank",
            kind="rerank",
            model="reranker-model",
            endpoint="https://r.test/v1/rerank",
        ),
    )
    assert clone is not None
    assert clone.kind == "rerank"
    assert clone.model == "reranker-model"
    # api_key 省略 → 复用源 key。
    assert clone.api_key_last4 == "1234"
    await engine.dispose()


async def test_duplicate_pure_copy_conflicts():
    """不覆盖任何区分字段的纯复制会撞 (kind, model, api_key_hash)。"""
    engine, store = await _store()
    src = await store.create(_create())
    with pytest.raises(CredentialConflictError):
        await store.duplicate(src.id, CredentialDuplicate(name="dup"))
    await engine.dispose()


async def test_same_key_allowed_across_kinds():
    engine, store = await _store()
    await store.create(_create(kind="embed", api_key="sk-shared"))
    # 唯一约束是 (kind, model, api_key_hash)：同一把 key 换 kind 不冲突。
    record = await store.create(
        _create(
            kind="llm_rerank",
            name="llm",
            api_key="sk-shared",
            endpoint="https://api.test/v1",
            model="chat-model",
        )
    )
    assert record.kind == "llm_rerank"
    assert record.api_key_last4 == "ared"
    await engine.dispose()


async def test_same_key_same_kind_allowed_across_models():
    """放宽后：同 kind + 同 key + 不同 model 允许并存（主/备模型）。"""
    engine, store = await _store()
    await store.create(_create(kind="embed", model="model-a", api_key="sk-one"))
    record = await store.create(
        _create(kind="embed", name="backup", model="model-b", api_key="sk-one")
    )
    assert record.model == "model-b"
    await engine.dispose()


async def test_duplicate_missing_source_returns_none():
    engine, store = await _store()
    assert (
        await store.duplicate(999, CredentialDuplicate(name="x", api_key="sk-x"))
        is None
    )
    await engine.dispose()


async def test_create_duplicate_key_conflicts():
    engine, store = await _store()
    await store.create(_create(api_key="sk-same"))
    # 同 kind + 同 model + 同 key 三者全同才算重复行。
    with pytest.raises(CredentialConflictError):
        await store.create(_create(name="other", api_key="sk-same"))
    await engine.dispose()


def test_hash_key_is_sha256_hex():
    assert _hash_key("abc") == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
