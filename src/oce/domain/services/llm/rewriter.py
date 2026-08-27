"""Query rewriter using LLM for improving recall.

将用户查询改写为多个不同角度的查询，特别是解决中文查询 vs 英文文件名的问题。
"""
from __future__ import annotations

from loguru import logger

from oce.domain.services.llm.client import LLMClient
from oce.domain.services.llm.prompts import REWRITE_PROMPT_TEMPLATE

# prompt 脚手架片段。小参数模型会把指令原样回显，这些文本一旦被当成
# 查询送进检索会污染召回，必须在解析阶段拦掉。
_PROMPT_ECHO_MARKERS = (
    "改写策略",
    "用户查询",
    "改写后的查询",
    "搜索关键词",
    "召回率",
    "每行一个",
    "不要编号",
    "文件名版本",
    "英文关键词版本",
    "功能描述版本",
    "查询改写助手",
    "要求:",
)

# 改写结果是搜索关键词，超长说明模型在输出解释或指令
_MAX_REWRITE_CHARS = 80


class QueryRewriter:
    """基于 LLM 的查询改写器，生成多角度查询提升召回率。"""

    def __init__(
        self,
        client: LLMClient,
        model: str = "deepseek-v4-flash",
        num_rewrites: int = 3,
    ):
        """
        Args:
            client: LLM 客户端
            model: 模型名称
            num_rewrites: 生成的改写查询数量
        """
        self.client = client
        self.model = model
        self.num_rewrites = num_rewrites

    async def rewrite(self, query: str) -> list[str]:
        """
        将查询改写为多个版本。

        Args:
            query: 原始查询

        Returns:
            改写后的查询列表（包含原查询）
        """
        if not query or not query.strip():
            return [query]

        logger.info(f"Query rewrite: original='{query}'")

        try:
            rewritten_queries = await self._llm_rewrite(query)
            
            # 确保原查询在列表中
            if query not in rewritten_queries:
                rewritten_queries.insert(0, query)
            
            logger.info(f"Query rewrite: generated {len(rewritten_queries)} queries: {rewritten_queries}")
            return rewritten_queries

        except Exception as e:
            logger.warning(f"Query rewrite failed: {e}, using original query only")
            return [query]

    def _is_valid_rewrite(self, candidate: str) -> bool:
        """判断一行输出是否为可用的改写查询。

        小参数模型会把 prompt 指令原样回显，这些文本若混进检索会污染召回，
        因此按长度和脚手架关键词双重拦截。
        """
        if len(candidate) > _MAX_REWRITE_CHARS:
            return False
        if len(candidate) < 3:
            return False
        return not any(marker in candidate for marker in _PROMPT_ECHO_MARKERS)

    async def _llm_rewrite(self, query: str) -> list[str]:
        """
        调用 LLM 进行查询改写。

        Returns:
            改写后的查询列表
        """
        prompt = REWRITE_PROMPT_TEMPLATE.format(
            num_rewrites=self.num_rewrites, query=query
        )

        messages = [{"role": "user", "content": prompt}]

        response = await self.client.chat(
            messages,
            model=self.model,
            temperature=0.2,
        )

        # 解析行分隔响应
        rewritten_queries: list[str] = []
        rejected: list[str] = []

        for line in response.strip().split("\n"):
            # 清理：去除编号、markdown、多余空格
            cleaned = line.strip()
            # 移除可能的编号前缀（1. 或 - 或 * 等）
            cleaned = cleaned.lstrip("0123456789.-*• \t")
            if not cleaned:
                continue

            if self._is_valid_rewrite(cleaned):
                rewritten_queries.append(cleaned)
            else:
                rejected.append(cleaned)

        # 限制数量
        rewritten_queries = rewritten_queries[: self.num_rewrites]

        if rejected:
            logger.warning(f"Query rewrite dropped {len(rejected)} invalid lines: {rejected}")

        if not rewritten_queries:
            logger.warning(f"LLM returned no valid queries. Raw response: {response[:200]}")

        return rewritten_queries
