"""Chain 聚合根 - 工作集抽象

Chain 是客户端工作集的领域抽象，职责：
- 管理 Blob 成员列表
- 实现 Checkpoint 版本控制
- 支持增量更新（added/deleted）

不变量：
- chain_id 必须是有效的 UUID
- version 必须单调递增
- members 集合去重
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Chain:
    """Chain 聚合根 - 工作集"""
    
    chain_id: str                         # UUID
    version: int                          # 版本号（从 1 开始）
    members: set[str] = field(default_factory=set)  # Blob 成员集合
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        """验证不变量"""
        if not self._is_valid_uuid(self.chain_id):
            raise ValueError(f"Invalid chain_id (not UUID): {self.chain_id}")
        if self.version < 1:
            raise ValueError(f"Invalid version (must be >= 1): {self.version}")
    
    @staticmethod
    def _is_valid_uuid(s: str) -> bool:
        """验证 UUID 格式"""
        try:
            uuid.UUID(s, version=4)
            return True
        except (ValueError, AttributeError):
            return False
    
    @classmethod
    def create(cls, members: list[str]) -> Chain:
        """创建新 Chain"""
        return cls(
            chain_id=str(uuid.uuid4()),
            version=1,
            members=set(members),
        )
    
    def apply_checkpoint(
        self,
        added: list[str],
        deleted: list[str],
    ) -> None:
        """应用 Checkpoint（增量更新）
        
        操作：
        1. members ∪ added
        2. members - deleted
        3. version += 1
        """
        # 添加新成员
        for blob_name in added:
            self.members.add(blob_name)
        
        # 删除成员
        for blob_name in deleted:
            self.members.discard(blob_name)
        
        # 版本递增
        self.version += 1
        self.updated_at = datetime.now(timezone.utc)
    
    def contains(self, blob_name: str) -> bool:
        """判断是否包含 Blob"""
        return blob_name in self.members
    
    def size(self) -> int:
        """获取成员数量"""
        return len(self.members)
    
    def is_empty(self) -> bool:
        """判断是否为空"""
        return len(self.members) == 0
    
    def get_checkpoint_token(self) -> str:
        """获取 Checkpoint 令牌
        
        格式：{chain_id}:{version}
        不透明令牌，客户端只需存储并回传
        """
        return f"{self.chain_id}:{self.version}"
    
    @staticmethod
    def parse_checkpoint_token(token: str) -> tuple[str, int] | None:
        """解析 Checkpoint 令牌
        
        返回：(chain_id, version) 或 None（格式非法）
        """
        if not token or ":" not in token:
            return None
        
        chain_id, _, version_str = token.rpartition(":")
        if not chain_id or not version_str.isdigit():
            return None
        
        # 验证 UUID 格式
        if not Chain._is_valid_uuid(chain_id):
            return None
        
        return chain_id, int(version_str)
    
    def __repr__(self) -> str:
        return (
            f"Chain(id={self.chain_id[:8]}..., "
            f"version={self.version}, members={len(self.members)})"
        )
