"""Chain 聚合根单元测试"""

import pytest

from oce.domain.chain.chain import Chain
from tests.conftest import make_chain, make_sha256


class TestChainCreation:
    """测试 Chain 创建"""
    
    def test_create_chain_with_factory(self):
        """使用工厂方法创建 Chain"""
        members = [make_sha256("file1"), make_sha256("file2")]
        
        chain = Chain.create(members)
        
        assert chain.chain_id is not None
        assert chain.version == 1
        assert len(chain.members) == 2
    
    def test_create_chain_with_invalid_uuid(self):
        """创建 Chain 时 chain_id 格式非法应抛异常"""
        with pytest.raises(ValueError, match="Invalid chain_id"):
            Chain(
                chain_id="not-a-uuid",
                version=1,
            )


class TestCheckpointOperations:
    """测试 Checkpoint 操作"""
    
    def test_apply_checkpoint_add_members(self):
        """应用 Checkpoint - 添加成员"""
        chain = make_chain(members=[make_sha256("file1")])
        initial_version = chain.version
        
        new_blob = make_sha256("file2")
        chain.apply_checkpoint(added=[new_blob], deleted=[])
        
        assert new_blob in chain.members
        assert chain.version == initial_version + 1
    
    def test_apply_checkpoint_delete_members(self):
        """应用 Checkpoint - 删除成员"""
        blob1 = make_sha256("file1")
        blob2 = make_sha256("file2")
        chain = make_chain(members=[blob1, blob2])
        
        chain.apply_checkpoint(added=[], deleted=[blob1])
        
        assert blob1 not in chain.members
        assert blob2 in chain.members
    
    def test_apply_checkpoint_mixed_operations(self):
        """应用 Checkpoint - 混合操作"""
        blob1 = make_sha256("file1")
        blob2 = make_sha256("file2")
        blob3 = make_sha256("file3")
        
        chain = make_chain(members=[blob1, blob2])
        chain.apply_checkpoint(added=[blob3], deleted=[blob1])
        
        assert blob1 not in chain.members
        assert blob2 in chain.members
        assert blob3 in chain.members


class TestCheckpointToken:
    """测试 Checkpoint 令牌"""
    
    def test_get_checkpoint_token(self):
        """获取 Checkpoint 令牌"""
        chain = make_chain()
        
        token = chain.get_checkpoint_token()
        
        assert ":" in token
        assert token.startswith(chain.chain_id)
        assert token.endswith(str(chain.version))
    
    def test_parse_valid_checkpoint_token(self):
        """解析合法 Checkpoint 令牌"""
        chain = make_chain()
        token = chain.get_checkpoint_token()
        
        parsed = Chain.parse_checkpoint_token(token)
        
        assert parsed is not None
        chain_id, version = parsed
        assert chain_id == chain.chain_id
        assert version == chain.version
    
    def test_parse_invalid_checkpoint_token(self):
        """解析非法 Checkpoint 令牌"""
        invalid_tokens = [
            "",
            "no-colon",
            "invalid-uuid:1",
            "valid-uuid-format-but-not-uuid:not-a-number",
        ]
        
        for token in invalid_tokens:
            assert Chain.parse_checkpoint_token(token) is None


class TestChainQueries:
    """测试 Chain 查询方法"""
    
    def test_contains(self):
        """测试 contains 方法"""
        blob = make_sha256("file1")
        chain = make_chain(members=[blob])
        
        assert chain.contains(blob)
        assert not chain.contains(make_sha256("other"))
    
    def test_size(self):
        """测试 size 方法"""
        chain = make_chain(members=[make_sha256("f1"), make_sha256("f2")])
        
        assert chain.size() == 2
    
    def test_is_empty(self):
        """测试 is_empty 方法"""
        empty_chain = make_chain(members=[])
        non_empty_chain = make_chain(members=[make_sha256("file1")])
        
        assert empty_chain.is_empty()
        assert not non_empty_chain.is_empty()
