"""Milvus 3.0 集成测试

使用 Milvus Lite（本地文件数据库）进行真实测试。
无需 Docker，数据存储在 ./test_data/milvus.db
"""

import pytest
from pathlib import Path
import shutil

from oce.shared.config.settings import MilvusSettings
from oce.infrastructure.milvus3 import Milvus3Client, Milvus3SearchStore


@pytest.fixture(scope="module")
def test_db_path(tmp_path_factory):
    """创建临时测试数据库目录"""
    db_dir = tmp_path_factory.mktemp("milvus_test")
    db_file = db_dir / "milvus.db"
    yield str(db_file)

    # 测试结束后清理（忽略 Windows 文件锁错误）
    try:
        import time
        time.sleep(0.5)  # 等待 Milvus Lite 释放文件
        if db_dir.exists():
            shutil.rmtree(db_dir, ignore_errors=True)
    except Exception:
        pass  # 忽略清理错误


@pytest.fixture(scope="module")
def milvus_settings(test_db_path):
    """Milvus Lite 配置（本地文件数据库）"""
    return MilvusSettings(
        endpoint=test_db_path,  # Milvus Lite：直接传文件路径
        token=None,
        collection_name="test_oce_chunks",
        dense_dim=128,  # 使用小维度加快测试
    )


@pytest.fixture
async def milvus_client(milvus_settings):
    """初始化 Milvus 客户端（每个测试独立）"""
    client = Milvus3Client(milvus_settings)

    await client.initialize()

    # 清空已有数据（确保测试隔离）
    try:
        await client._client.delete(
            collection_name=milvus_settings.collection_name,
            filter="content_hash != ''",  # 删除所有数据
        )
    except Exception:
        pass  # Collection 可能是空的

    yield client
    await client.close()


@pytest.mark.integration
@pytest.mark.asyncio
class TestMilvus3Integration:
    """Milvus 3.0 集成测试（使用 Milvus Lite）"""
    
    async def test_insert_and_search(self, milvus_client):
        """测试完整的插入和检索流程"""
        # 插入测试数据
        chunks = [
            {
                "chunk_id": "1" * 64,
                "content_hash": "hash1",
                "content": "def calculate_sum(a, b): return a + b",
                "embedding": [0.1] * 128,
                "blob_name": "a" * 64,
                "metadata": {"path": "src/math.py", "start_line": 1, "end_line": 1},
            },
            {
                "chunk_id": "2" * 64,
                "content_hash": "hash2",
                "content": "def calculate_product(a, b): return a * b",
                "embedding": [0.2] * 128,
                "blob_name": "a" * 64,
                "metadata": {"path": "src/math.py", "start_line": 3, "end_line": 3},
            },
            {
                "chunk_id": "3" * 64,
                "content_hash": "hash3",
                "content": "class Calculator: pass",
                "embedding": [0.3] * 128,
                "blob_name": "b" * 64,
                "metadata": {"path": "src/calculator.py", "start_line": 1, "end_line": 1},
            },
        ]
        
        result = await milvus_client.insert(chunks)
        assert result["inserted"] == 3
        
        # dense 向量检索
        results = await milvus_client.search(
            query_text="calculate sum function",
            query_embedding=[0.15] * 128,  # 接近 hash1
            blob_filter=["a" * 64],
            top_k=2,
        )
        
        assert len(results) >= 1
        assert results[0]["content_hash"] in ["hash1", "hash2"]
        assert results[0]["blob_name"] == "a" * 64
    
    async def test_blob_filter(self, milvus_client):
        """测试 blob_name 过滤"""
        # 插入测试数据（两个不同的 blob）
        chunks = [
            {
                "chunk_id": "4" * 64,
                "content_hash": "filter_hash1",
                "content": "class Calculator: pass",
                "embedding": [0.3] * 128,
                "blob_name": "b" * 64,
                "metadata": {"path": "src/calculator.py"},
            },
            {
                "chunk_id": "5" * 64,
                "content_hash": "filter_hash2",
                "content": "def add(a, b): return a + b",
                "embedding": [0.2] * 128,
                "blob_name": "a" * 64,
                "metadata": {"path": "src/math.py"},
            },
        ]
        await milvus_client.insert(chunks)

        # 只搜索 calculator.py
        results = await milvus_client.search(
            query_text="calculator",
            query_embedding=[0.3] * 128,
            blob_filter=["b" * 64],
            top_k=10,
        )

        assert len(results) >= 1
        for result in results:
            assert result["blob_name"] == "b" * 64

    async def test_delete_by_blob(self, milvus_client):
        """测试按 blob 删除"""
        # 插入测试数据
        chunks = [
            {
                "chunk_id": "6" * 64,
                "content_hash": "delete_hash1",
                "content": "class ToDelete: pass",
                "embedding": [0.4] * 128,
                "blob_name": "c" * 64,
                "metadata": {"path": "src/delete_me.py"},
            },
        ]
        await milvus_client.insert(chunks)

        # 删除
        deleted = await milvus_client.delete_by_blob("c" * 64)
        assert deleted >= 1

        # 验证删除后搜索不到
        results = await milvus_client.search(
            query_text="delete",
            query_embedding=[0.4] * 128,
            blob_filter=["c" * 64],
            top_k=10,
        )

        assert len(results) == 0


@pytest.mark.integration
@pytest.mark.asyncio
class TestMilvus3SearchStoreIntegration:
    """Milvus3SearchStore 集成测试"""
    
    @pytest.fixture
    async def search_store(self, milvus_settings):
        """初始化 SearchStore"""
        store = Milvus3SearchStore(milvus_settings)
        await store._ensure_initialized()
        yield store
        await store.close()
    
    async def test_search_store_upsert_and_search(self, search_store):
        """测试 SearchStore 的 upsert 和 search"""
        # Upsert 数据
        items = [
            {
                "chunk_id": "test_hash_1",
                "blob_name": "d" * 64,
                "vector": [0.5] * 128,
                "content": "def test_function(): pass",
                "metadata": {"path": "test.py"},
            }
        ]
        
        await search_store.upsert(items)
        
        # 搜索
        results = await search_store.search(
            query="test function",
            query_vector=[0.5] * 128,
            allowed_blob_names=["d" * 64],
            top_k=5,
            vector_threshold=0.0,
        )
        
        assert len(results) >= 1
        assert results[0].blob_name == "d" * 64
        assert results[0].path == "test.py"
