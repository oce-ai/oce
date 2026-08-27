"""Milvus 3.0 集成单元测试

测试 Milvus 3.0 客户端、Schema 和 SearchStore 的基本功能。
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from pymilvus.client.types import LoadState

from oce.shared.config.settings import MilvusSettings
from oce.infrastructure.milvus3 import (
    create_oce_collection_schema,
    Milvus3Client,
    Milvus3SearchStore,
)


class TestMilvusSchema:
    """测试 Collection Schema 定义"""

    def test_create_schema_default_dim(self):
        """测试默认维度的 Schema 创建"""
        schema = create_oce_collection_schema()

        # 验证是 CollectionSchema 对象
        from pymilvus import CollectionSchema
        assert isinstance(schema, CollectionSchema)

        # 验证字段数量
        assert len(schema.fields) == 6

        # 验证关键字段
        field_names = [f.name for f in schema.fields]
        assert "chunk_id" in field_names
        assert "content_hash" in field_names
        assert "content" in field_names
        assert "dense_vector" in field_names
        assert "blob_name" in field_names
        assert "metadata" in field_names
        assert len(schema.functions) == 0

    def test_create_schema_custom_dim(self):
        """测试自定义维度的 Schema 创建"""
        schema = create_oce_collection_schema(dense_dim=768)

        dense_field = next(f for f in schema.fields if f.name == "dense_vector")
        assert dense_field.params["dim"] == 768


class TestMilvus3Client:
    """测试 Milvus 3.0 客户端"""

    @pytest.fixture
    def mock_settings(self):
        """Mock Milvus 配置"""
        return MilvusSettings(
            endpoint="http://localhost:19530",
            token=None,
            collection_name="test_collection",
            dense_dim=1024,
        )

    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.client.AsyncMilvusClient")
    async def test_client_init_creates_collection(self, mock_milvus_client_class, mock_settings):
        """测试客户端初始化时自动创建 Collection"""
        mock_client = Mock()
        mock_client.has_collection = AsyncMock(return_value=False)
        mock_client.create_collection = AsyncMock()
        mock_client.create_index = AsyncMock()
        mock_client.load_collection = AsyncMock()
        mock_client.prepare_index_params = Mock(return_value=Mock())
        mock_milvus_client_class.return_value = mock_client

        client = Milvus3Client(mock_settings)
        await client.initialize()

        # 验证连接参数
        mock_milvus_client_class.assert_called_once_with(
            uri=mock_settings.endpoint,
            token=mock_settings.token,
        )

        # 验证 Collection 创建
        assert mock_client.create_collection.called

    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.client.MilvusClient")
    async def test_local_uri_uses_sync_client_in_worker_thread(
        self,
        mock_milvus_client_class,
    ):
        settings = MilvusSettings(endpoint="./oce_milvus.db", collection_name="test_collection")
        mock_client = Mock()
        mock_client.has_collection.return_value = True
        mock_client.get_load_state.return_value = {"state": LoadState.Loaded}
        mock_milvus_client_class.return_value = mock_client

        client = Milvus3Client(settings)
        await client.initialize()

        mock_milvus_client_class.assert_called_once_with(uri=settings.endpoint, token=settings.token)
        mock_client.has_collection.assert_called_once_with(settings.collection_name)
        mock_client.get_load_state.assert_called_once_with(settings.collection_name)

    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.client.AsyncMilvusClient")
    async def test_client_init_loads_existing_collection(self, mock_milvus_client_class, mock_settings):
        """测试客户端初始化时加载已存在的 Collection"""
        mock_client = Mock()
        mock_client.has_collection = AsyncMock(return_value=True)
        mock_client.get_load_state = AsyncMock(return_value={"state": LoadState.NotLoad})
        mock_client.load_collection = AsyncMock()
        mock_milvus_client_class.return_value = mock_client

        client = Milvus3Client(mock_settings)
        await client.initialize()

        # 验证加载 Collection
        mock_client.load_collection.assert_called_once_with(mock_settings.collection_name)

    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.client.AsyncMilvusClient")
    async def test_insert_chunks(self, mock_milvus_client_class, mock_settings):
        """测试插入向量数据"""
        mock_client = Mock()
        mock_client.has_collection = AsyncMock(return_value=True)
        mock_client.get_load_state = AsyncMock(return_value={"state": LoadState.Loaded})
        mock_client.upsert = AsyncMock(return_value={"upsert_count": 2})
        mock_milvus_client_class.return_value = mock_client

        client = Milvus3Client(mock_settings)
        await client.initialize()
        
        chunks = [
            {
                "chunk_id": "chunk1",
                "content_hash": "hash1",
                "content": "def main():",
                "embedding": [0.1] * 1024,
                "blob_name": "test.py",
                "metadata": {"path": "test.py", "start_line": 1, "end_line": 1},
            },
            {
                "chunk_id": "chunk2",
                "content_hash": "hash2",
                "content": "print('hello')",
                "embedding": [0.2] * 1024,
                "blob_name": "test.py",
                "metadata": {"path": "test.py", "start_line": 2, "end_line": 2},
            },
        ]
        
        result = await client.insert(chunks)
        
        # 验证插入调用
        assert mock_client.upsert.called
        assert result["inserted"] == 2

    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.client.AsyncMilvusClient")
    async def test_insert_limits_content_by_utf8_bytes(
        self,
        mock_milvus_client_class,
        mock_settings,
    ):
        mock_client = Mock()
        mock_client.has_collection = AsyncMock(return_value=True)
        mock_client.get_load_state = AsyncMock(return_value={"state": LoadState.Loaded})
        mock_client.upsert = AsyncMock(return_value={"upsert_count": 1})
        mock_milvus_client_class.return_value = mock_client
        client = Milvus3Client(mock_settings)
        await client.initialize()

        original = "界" * 30_000
        await client.insert([
            {
                "chunk_id": "chunk1",
                "content_hash": "hash1",
                "content": original,
                "embedding": [0.1] * 1024,
                "blob_name": "large.svg",
                "metadata": {"path": "large.svg"},
            }
        ])

        row = mock_client.upsert.await_args.kwargs["data"][0]
        assert len(row["content"].encode("utf-8")) <= 65_535
        assert row["content"].encode("utf-8").decode("utf-8") == row["content"]
        assert row["metadata"]["content_truncated"] is True
        assert row["metadata"]["content_bytes"] == len(original.encode("utf-8"))
    
    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.client.AsyncMilvusClient")
    async def test_dense_search(self, mock_milvus_client_class, mock_settings):
        """测试 dense 向量检索和候选查询参数。"""
        mock_client = Mock()
        mock_client.has_collection = AsyncMock(return_value=True)
        mock_client.get_load_state = AsyncMock(return_value={"state": LoadState.Loaded})
        hit = Mock()
        hit.entity = {
            "content_hash": "hash1",
            "content": "def main():",
            "blob_name": "a" * 64,
            "metadata": {"path": "src/main.py"},
        }
        hit.distance = 0.9
        mock_client.search = AsyncMock(return_value=[[hit]])
        mock_milvus_client_class.return_value = mock_client

        client = Milvus3Client(mock_settings)
        await client.initialize()

        results = await client.search(
            query_text="main function",
            query_embedding=[0.1] * 1024,
            blob_filter=["a" * 64],
            top_k=10,
        )

        assert len(results) == 1
        assert results[0]["content_hash"] == "hash1"
        assert results[0]["score"] == 0.9
        mock_client.search.assert_awaited_once()
        search_kwargs = mock_client.search.await_args.kwargs
        assert search_kwargs["limit"] == 10
        assert search_kwargs["search_params"]["params"]["ef"] == mock_settings.hnsw_ef_search
        assert search_kwargs["filter"] == f'blob_name in ["{"a" * 64}"]'

    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.client.MilvusClient")
    async def test_local_search_parses_dict_hits(self, mock_milvus_client_class):
        settings = MilvusSettings(endpoint="./oce_milvus.db", collection_name="test_collection")
        mock_client = Mock()
        mock_client.has_collection.return_value = True
        mock_client.get_load_state.return_value = Mock(name="Loaded")
        mock_client.search.return_value = [[{
            "entity": {
                "content_hash": "hash1",
                "content": "def main():",
                "blob_name": "a" * 64,
                "metadata": {"path": "src/main.py"},
            },
            "distance": 0.9,
        }]]
        mock_milvus_client_class.return_value = mock_client

        client = Milvus3Client(settings)
        await client.initialize()
        results = await client.search("main", [0.1] * 1024, top_k=1)

        assert results[0]["content_hash"] == "hash1"
        assert results[0]["score"] == 0.9

    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.client.AsyncMilvusClient")
    async def test_dense_search_raises_ef_to_candidate_limit(
        self,
        mock_milvus_client_class,
        mock_settings,
    ):
        mock_client = Mock()
        mock_client.has_collection = AsyncMock(return_value=True)
        mock_client.get_load_state = AsyncMock(return_value=Mock(name="Loaded"))
        mock_client.search = AsyncMock(return_value=[[]])
        mock_milvus_client_class.return_value = mock_client
        client = Milvus3Client(mock_settings)

        await client.search(
            query_text="query",
            query_embedding=[0.1] * 1024,
            top_k=50,
        )

        search_kwargs = mock_client.search.await_args.kwargs
        assert search_kwargs["limit"] == 50
        assert search_kwargs["search_params"]["params"]["ef"] == max(
            mock_settings.hnsw_ef_search,
            50 * 2,
        )

    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.client.AsyncMilvusClient")
    async def test_blob_filters_reject_non_sha256_values(
        self,
        mock_milvus_client_class,
        mock_settings,
    ):
        mock_milvus_client_class.return_value = Mock()
        client = Milvus3Client(mock_settings)

        with pytest.raises(ValueError, match="SHA256"):
            await client.search(
                query_text="query",
                query_embedding=[0.1] * 1024,
                blob_filter=['x" or true'],
            )
        with pytest.raises(ValueError, match="SHA256"):
            await client.delete_by_blob('x" or true')

    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.client.AsyncMilvusClient")
    async def test_delete_by_blob_uses_validated_filter(
        self,
        mock_milvus_client_class,
        mock_settings,
    ):
        mock_client = Mock()
        mock_client.delete = AsyncMock(return_value=["chunk1", "chunk2"])
        mock_milvus_client_class.return_value = mock_client
        client = Milvus3Client(mock_settings)
        blob_name = "a" * 64

        deleted = await client.delete_by_blob(blob_name)

        assert deleted == 2
        mock_client.delete.assert_awaited_once_with(
            collection_name=mock_settings.collection_name,
            filter=f'blob_name == "{blob_name}"',
        )


class TestMilvus3SearchStore:
    """测试 SearchStore 实现"""
    
    @pytest.fixture
    def mock_settings(self):
        return MilvusSettings(
            endpoint="http://localhost:19530",
            collection_name="test_collection",
        )
    
    @patch("oce.infrastructure.milvus3.search_store.Milvus3Client")
    def test_search_store_init(self, mock_client_class, mock_settings):
        """测试 SearchStore 初始化"""
        store = Milvus3SearchStore(mock_settings)
        
        # 验证客户端创建
        mock_client_class.assert_called_once_with(mock_settings)

    @pytest.mark.asyncio
    @patch("oce.infrastructure.milvus3.search_store.Milvus3Client")
    async def test_search_passes_scope_to_milvus_before_ranking(
        self,
        mock_client_class,
        mock_settings,
    ):
        mock_client = mock_client_class.return_value
        mock_client.initialize = AsyncMock()
        mock_client.search = AsyncMock(return_value=[{
            "content_hash": "hash1",
            "content": "def main():",
            "blob_name": "a" * 64,
            "metadata": {"path": "src/main.py", "start_line": 1, "end_line": 2},
            "score": 0.9,
        }])
        store = Milvus3SearchStore(mock_settings)

        hits = await store.search(
            query="main function",
            query_vector=[0.1] * 1024,
            allowed_blob_names=["a" * 64],
            top_k=5,
        )

        assert [hit.path for hit in hits] == ["src/main.py"]
        assert mock_client.search.await_args.kwargs["blob_filter"] == ["a" * 64]
        assert mock_client.search.await_args.kwargs["top_k"] == 5
