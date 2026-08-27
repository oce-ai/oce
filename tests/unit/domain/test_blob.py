"""Blob 聚合根单元测试"""

import pytest

from oce.domain.blob.blob import Blob, BlobStatus
from tests.conftest import make_blob, make_chunk_ref, make_sha256


class TestBlobCreation:
    """测试 Blob 创建"""
    
    def test_create_blob_with_valid_data(self):
        """创建合法 Blob"""
        blob = make_blob(
            blob_name=make_sha256("test"),
            path="src/test.py",
        )
        
        assert blob.blob_name is not None
        assert blob.path == "src/test.py"
        assert blob.status == BlobStatus.PENDING
        assert len(blob.chunks) == 0
    
    def test_create_blob_with_invalid_blob_name(self):
        """创建 Blob 时 blob_name 格式非法应抛异常"""
        with pytest.raises(ValueError, match="Invalid blob_name"):
            Blob(
                blob_name="not-a-sha256",
                path="src/test.py",
            )


class TestBlobChunkManagement:
    """测试 Chunk 管理"""
    
    def test_add_chunk(self):
        """添加 Chunk 引用"""
        blob = make_blob()
        chunk_ref = make_chunk_ref()
        
        blob.add_chunk(chunk_ref)
        
        assert len(blob.chunks) == 1
        assert blob.chunks[0] == chunk_ref
    
    def test_add_duplicate_chunk(self):
        """添加重复 Chunk 不会重复"""
        blob = make_blob()
        chunk_ref = make_chunk_ref()
        
        blob.add_chunk(chunk_ref)
        blob.add_chunk(chunk_ref)
        
        assert len(blob.chunks) == 1


class TestBlobStatusTransition:
    """测试状态转换"""
    
    def test_mark_ready_with_chunks(self):
        """有 chunks 时可以标记为 ready"""
        blob = make_blob()
        blob.add_chunk(make_chunk_ref())
        
        blob.mark_ready()
        
        assert blob.status == BlobStatus.READY
        assert blob.error_message is None
    
    def test_mark_ready_without_chunks_supports_empty_files(self):
        blob = make_blob()

        blob.mark_ready()

        assert blob.status == BlobStatus.READY
    
    def test_mark_error(self):
        """标记为错误状态"""
        blob = make_blob()
        error_msg = "Embedding failed"
        
        blob.mark_error(error_msg)
        
        assert blob.status == BlobStatus.ERROR
        assert blob.error_message == error_msg


class TestBlobExpiration:
    """测试过期判定"""
    
    def test_blob_not_expired_within_ttl(self, freezed_time):
        """TTL 内不过期"""
        blob = make_blob()
        blob.last_seen = freezed_time  # 当前时间
        
        assert not blob.is_expired(ttl_days=30)
    
    def test_blob_expired_beyond_ttl(self, freezed_time):
        """超过 TTL 则过期"""
        from datetime import timedelta
        
        blob = make_blob()
        blob.last_seen = freezed_time - timedelta(days=31)  # 31 天前
        
        assert blob.is_expired(ttl_days=30)


class TestBlobTouch:
    """测试 touch 操作"""
    
    def test_touch_updates_last_seen(self, freezed_time):
        """touch 更新 last_seen"""
        from datetime import timedelta
        
        blob = make_blob()
        old_time = freezed_time - timedelta(hours=1)
        blob.last_seen = old_time
        
        blob.touch()
        
        assert blob.last_seen > old_time
