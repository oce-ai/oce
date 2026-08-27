"""统一的 LLM 客户端协议。

三个 LLM 消费者（reranker / rewriter / intent classifier）共享同一个 chat 协议。
此前该协议在 llm_reranker 与 query_rewriter 中各定义一份，此处收敛为单一来源，
避免重复。基础设施层的 OpenAI 兼容客户端实现它。
"""
from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """LLM 聊天客户端协议。

    额外参数（model / temperature / max_tokens 等）经 kwargs 透传给具体实现，
    以适配 rerank、query rewrite、intent classify 各自不同的调用参数。
    """

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """发送聊天请求，返回模型响应内容。"""
        ...
