"""查询意图分类器 - LLM-based 7-category classifier

意图分类：
- S (SYMBOL): 符号定义查询
- C (CALL_CHAIN): 调用链/流程查询  
- R (REFERENCE): 引用/使用位置查询
- P (PATH): 文件路径查询
- F (FEATURE): 功能实现查询
- O (OVERVIEW): 架构/机制概览查询
- M (COMPOUND): 多条件复合查询
"""

from __future__ import annotations

from enum import Enum

from oce.domain.services.llm.client import LLMClient
from oce.domain.services.llm.prompts import INTENT_SYSTEM_PROMPT, INTENT_USER_TEMPLATE


class QueryIntent(str, Enum):
    """查询意图枚举"""
    
    SYMBOL = "S"          # 符号定义
    CALL_CHAIN = "C"      # 调用链
    REFERENCE = "R"       # 引用位置
    PATH = "P"            # 文件路径
    FEATURE = "F"         # 功能实现
    OVERVIEW = "O"        # 架构概览
    COMPOUND = "M"        # 复合查询


# Label 映射
LABEL_TO_INTENT = {
    'S': QueryIntent.SYMBOL,
    'C': QueryIntent.CALL_CHAIN,
    'R': QueryIntent.REFERENCE,
    'P': QueryIntent.PATH,
    'F': QueryIntent.FEATURE,
    'O': QueryIntent.OVERVIEW,
    'M': QueryIntent.COMPOUND,
}


class IntentClassifier:
    """查询意图分类器（LLM-based）"""
    
    def __init__(self, llm_client: LLMClient, model: str):
        """
        Args:
            llm_client: LLM 客户端（需支持 chat 方法）
            model: 模型名称
        """
        self.llm_client = llm_client
        self.model = model
    
    async def classify(self, query: str) -> QueryIntent:
        """分类查询意图
        
        Args:
            query: 查询文本
            
        Returns:
            QueryIntent 枚举值
        """
        user_prompt = INTENT_USER_TEMPLATE.format(query=query)
        messages = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        
        response = await self.llm_client.chat(
            messages=messages,
            model=self.model,
            temperature=0,
            max_tokens=2,
        )
        
        label = response.strip().upper()
        return LABEL_TO_INTENT.get(label, QueryIntent.FEATURE)
