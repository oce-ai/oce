"""Verify SQL repositories against an in-memory SQLite database."""

import pytest
from datetime import datetime, timezone

from oce.domain.blob.blob import Blob, BlobStatus
from oce.domain.chunk import Chunk, ChunkRef
from oce.infrastructure.persistence.sql_blob_repo import SqlBlobRepository
from oce.infrastructure.persistence.sql_chunk_repo import SqlChunkRepository
from oce.infrastructure.persistence.sql_exact_search_store import SqlExactSearchStore
from oce.infrastructure.persistence.sql_chain_repo import SqlChainRepository
from tests.conftest import make_sha256


@pytest.fixture
async def sqlite_session():
    """创建 SQLite 内存数据库 session"""
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from oce.infrastructure.persistence.models import (
        BlobModel,
        ChunkModel,
        BlobChunkModel,
        ChainModel,
        ChainMemberModel,
        SymbolOccurrenceModel,
    )
    from sqlalchemy.orm import declarative_base

    Base = declarative_base()

    # 复制新表定义
    class TestBlob(Base):
        __table__ = BlobModel.__table__.to_metadata(Base.metadata)

    class TestChunk(Base):
        __table__ = ChunkModel.__table__.to_metadata(Base.metadata)

    class TestBlobChunk(Base):
        __table__ = BlobChunkModel.__table__.to_metadata(Base.metadata)

    class TestChain(Base):
        __table__ = ChainModel.__table__.to_metadata(Base.metadata)

    class TestChainMember(Base):
        __table__ = ChainMemberModel.__table__.to_metadata(Base.metadata)

    class TestSymbolOccurrence(Base):
        __table__ = SymbolOccurrenceModel.__table__.to_metadata(Base.metadata)

    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session_factory() as session:
        yield session
        await session.rollback()

    await engine.dispose()


@pytest.mark.asyncio
async def test_blob_repository_crud(sqlite_session):
    """测试 BlobRepository CRUD 操作"""
    repo = SqlBlobRepository(sqlite_session)

    # 创建 Blob
    blob = Blob(
        blob_name=make_sha256("test1"),
        path="src/test.py",
        status=BlobStatus.PENDING,
        chunks=[
            ChunkRef(content_hash=make_sha256("chunk1"), start_line=1, end_line=10),
            ChunkRef(content_hash=make_sha256("chunk2"), start_line=11, end_line=20),
        ],
    )

    # 保存
    await repo.save(blob)
    await sqlite_session.commit()

    # 读取
    loaded = await repo.get(blob.blob_name)
    assert loaded is not None
    assert loaded.blob_name == blob.blob_name
    assert loaded.path == "src/test.py"
    assert loaded.status == BlobStatus.PENDING
    assert len(loaded.chunks) == 2

    # 判断存在
    exists = await repo.exists(blob.blob_name)
    assert exists is True

    # 更新状态
    loaded.mark_ready()
    await repo.save(loaded)
    await sqlite_session.commit()

    reloaded = await repo.get(blob.blob_name)
    assert reloaded.status == BlobStatus.READY

    # 删除
    await repo.delete(blob.blob_name)
    await sqlite_session.commit()

    deleted = await repo.get(blob.blob_name)
    assert deleted is None


@pytest.mark.asyncio
async def test_chunk_repository_crud(sqlite_session):
    """测试 ChunkRepository CRUD 操作"""
    repo = SqlChunkRepository(sqlite_session)

    # 创建 Chunk
    chunk = Chunk(
        content_hash=make_sha256("content1"),
        path="src/test.py",
        content="print('hello')",
        start_line=1,
        end_line=1,
    )

    # 保存
    await repo.save(chunk)
    await sqlite_session.commit()

    # 读取
    loaded = await repo.get(chunk.content_hash)
    assert loaded is not None
    assert loaded.content_hash == chunk.content_hash
    assert loaded.content == "print('hello')"


@pytest.mark.asyncio
async def test_exact_search_store_prefers_definitions_and_honors_scope(sqlite_session):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    blob_repo = SqlBlobRepository(sqlite_session)
    chunk_repo = SqlChunkRepository(sqlite_session)
    definition = Chunk(
        make_sha256("definition"),
        "src/commands/copilot.rs",
        "#[tauri::command]\npub async fn copilot_get_models() {}",
        10,
        11,
    )
    reference = Chunk(
        make_sha256("reference"),
        "src/api/copilot.ts",
        'invoke("copilot_get_models")',
        20,
        20,
    )
    outside = Chunk(
        make_sha256("outside"),
        "vendor/copilot.rs",
        "pub fn copilot_get_models() {}",
        1,
        1,
    )
    helper = Chunk(
        make_sha256("helper"),
        "src/config/copilot.rs",
        "pub async fn copilot_get_models() {}",
        30,
        30,
    )
    await chunk_repo.save_many([definition, reference, outside, helper])
    names = [
        make_sha256(value)
        for value in ("definition", "reference", "outside", "helper")
    ]
    await blob_repo.save_many(
        [
            Blob(names[0], definition.path, BlobStatus.READY, chunks=[definition.to_ref()]),
            Blob(names[1], reference.path, BlobStatus.READY, chunks=[reference.to_ref()]),
            Blob(names[2], outside.path, BlobStatus.READY, chunks=[outside.to_ref()]),
            Blob(names[3], helper.path, BlobStatus.READY, chunks=[helper.to_ref()]),
        ]
    )
    await sqlite_session.commit()
    factory = async_sessionmaker(
        sqlite_session.bind, class_=AsyncSession, expire_on_commit=False
    )
    store = SqlExactSearchStore(factory)

    hits = await store.search_exact(
        identifiers=["copilot_get_models"],
        allowed_blob_names=[names[0], names[1], names[3]],
        top_k=10,
    )

    assert [hit.path for hit in hits] == [definition.path, helper.path, reference.path]


@pytest.mark.asyncio
async def test_exact_search_store_skips_unbounded_and_large_scopes(sqlite_session):
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    factory = async_sessionmaker(
        sqlite_session.bind, class_=AsyncSession, expire_on_commit=False
    )
    store = SqlExactSearchStore(factory, max_scope_blobs=2)

    unbounded = await store.search_exact(identifiers=["target"], top_k=10)
    oversized = await store.search_exact(
        identifiers=["target"],
        allowed_blob_names=[make_sha256(str(index)) for index in range(3)],
        top_k=10,
    )

    assert unbounded == []
    assert oversized == []


@pytest.mark.asyncio
async def test_chain_repository_checkpoint(sqlite_session):
    """测试 ChainRepository checkpoint 操作"""
    repo = SqlChainRepository(sqlite_session)

    # 创建 Chain
    members = [make_sha256("blob1"), make_sha256("blob2")]
    chain = await repo.create(members)
    await sqlite_session.commit()

    assert chain.chain_id is not None
    assert chain.version == 1
    assert len(chain.members) == 2

    # 获取成员
    loaded_members = await repo.get_members(chain.chain_id)
    assert len(loaded_members) == 2
    assert set(members) == loaded_members

    # 应用 checkpoint
    new_version = await repo.apply_checkpoint(
        chain.chain_id,
        added=[make_sha256("blob3")],
        deleted=[make_sha256("blob1")],
    )
    await sqlite_session.commit()

    assert new_version == 2

    # 验证成员变更
    updated_members = await repo.get_members(chain.chain_id)
    assert len(updated_members) == 2
    assert make_sha256("blob2") in updated_members
    assert make_sha256("blob3") in updated_members
    assert make_sha256("blob1") not in updated_members


@pytest.mark.asyncio
async def test_chain_repository_batches_large_member_sets(sqlite_session):
    repo = SqlChainRepository(sqlite_session)
    members = [make_sha256(f"blob-{index}") for index in range(17_000)]

    chain = await repo.create(members)
    await sqlite_session.commit()

    assert len(await repo.get_members(chain.chain_id)) == len(members)


@pytest.mark.asyncio
async def test_batch_operations(sqlite_session):
    """测试批量操作"""
    blob_repo = SqlBlobRepository(sqlite_session)
    chunk_repo = SqlChunkRepository(sqlite_session)

    # 批量保存 Chunk
    chunks = [
        Chunk(
            content_hash=make_sha256(f"chunk{i}"),
            path="src/test.py",
            content=f"line {i}",
            start_line=i,
            end_line=i,
        )
        for i in range(1, 6)
    ]
    await chunk_repo.save_many(chunks)

    # 批量保存 Blob
    blobs = [
        Blob(
            blob_name=make_sha256(f"blob{i}"),
            path=f"src/file{i}.py",
            status=BlobStatus.PENDING,
            chunks=[ChunkRef(content_hash=make_sha256(f"chunk{i}"), start_line=i, end_line=i)],
        )
        for i in range(1, 4)
    ]
    await blob_repo.save_many(blobs)
    await sqlite_session.commit()

    # 批量读取
    blob_names = [make_sha256(f"blob{i}") for i in range(1, 4)]
    loaded_blobs = await blob_repo.get_many(blob_names)
    assert len(loaded_blobs) == 3

    # 批量判断存在
    exists_map = await blob_repo.exists_many(blob_names)
    assert all(exists_map.values())


@pytest.mark.asyncio
async def test_blob_delete_only_removes_unreferenced_chunks(sqlite_session):
    blob_repo = SqlBlobRepository(sqlite_session)
    chunk_repo = SqlChunkRepository(sqlite_session)
    shared = Chunk(make_sha256("shared"), "", "shared", 1, 1)
    unique = Chunk(make_sha256("unique"), "", "unique", 2, 2)
    await chunk_repo.save_many([shared, unique])
    first_name = make_sha256("first-blob")
    second_name = make_sha256("second-blob")
    await blob_repo.save_many(
        [
            Blob(
                first_name,
                "first.py",
                chunks=[shared.to_ref(), unique.to_ref()],
            ),
            Blob(second_name, "second.py", chunks=[shared.to_ref()]),
        ]
    )
    await sqlite_session.commit()

    await blob_repo.delete(first_name)
    await sqlite_session.commit()

    assert await chunk_repo.get(shared.content_hash) is not None
    assert await chunk_repo.get(unique.content_hash) is None
