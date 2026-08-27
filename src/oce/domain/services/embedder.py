"""Embedder 领域服务 - 文本向量化协议

可插拔：OpenAI 兼容服务 / 本地模型 / 别的 API 都实现此协议。
约束：embed_documents 与 embed_query 必须同 model + 同维度。
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    """嵌入器协议（异步）"""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量文档向量化，返回与输入等长的向量列表"""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """查询向量化（与文档使用同 model + dims）"""
        ...
