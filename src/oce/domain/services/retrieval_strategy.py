"""意图驱动的检索策略决策表

根据查询意图选择最优的检索策略组合
"""

from __future__ import annotations

from dataclasses import dataclass

from oce.domain.services.llm.intent import QueryIntent


@dataclass
class RetrievalStrategy:
    """检索策略配置"""
    
    enable_path_index: bool = False      # 启用路径索引
    enable_query_rewrite: bool = False   # 启用查询改写
    enable_llm_rerank: bool = False      # 启用 LLM 重排
    boost_definitions: bool = False      # 提升定义位置权重
    boost_docs: bool = False             # 提升文档权重
    max_chunks_per_path: int = 3         # 每个文件最多返回块数
    
    # 未来扩展
    enable_multi_hop: bool = False       # 启用多跳检索（调用链）
    enable_reference_graph: bool = False # 启用引用图（依赖分析）


# 决策表：意图 → 检索策略
STRATEGY_TABLE: dict[QueryIntent, RetrievalStrategy] = {
    # S (SYMBOL): 符号定义查询
    # 符号名应在正文中定位；路径语义会把同名引用、模型和 DAO 提到定义前面。
    QueryIntent.SYMBOL: RetrievalStrategy(
        enable_path_index=False,
        enable_query_rewrite=True,
        enable_llm_rerank=True,
        boost_definitions=True,
        max_chunks_per_path=2,  # 减少冗余
    ),
    
    # C (CALL_CHAIN): 调用链查询
    # 保留原查询中的方向和边界信息，交给 LLM 判断调用关系。
    QueryIntent.CALL_CHAIN: RetrievalStrategy(
        enable_path_index=False,
        enable_query_rewrite=False,
        enable_llm_rerank=True,
        max_chunks_per_path=3,
        enable_multi_hop=False,  # 多跳检索尚未实现
    ),
    
    # R (REFERENCE): 引用/使用位置查询
    # 策略：查询改写 + 中等块数
    QueryIntent.REFERENCE: RetrievalStrategy(
        enable_path_index=False,
        enable_query_rewrite=True,
        enable_llm_rerank=False,
        max_chunks_per_path=4,  # 多个引用位置
        enable_reference_graph=False,  # 引用图检索尚未实现
    ),
    
    # P (PATH): 文件路径查询
    # 文件语义改写补足中英文差异，路径索引负责召回，LLM 决定最终顺序。
    QueryIntent.PATH: RetrievalStrategy(
        enable_path_index=True,
        enable_query_rewrite=True,
        enable_llm_rerank=True,
        max_chunks_per_path=2,
    ),
    
    # F (FEATURE): 功能实现查询
    # 功能描述需要跨中英文术语召回，再由正文相关性确定实现文件。
    QueryIntent.FEATURE: RetrievalStrategy(
        enable_path_index=False,
        enable_query_rewrite=True,
        enable_llm_rerank=True,
        max_chunks_per_path=3,
    ),
    
    # O (OVERVIEW): 架构/机制概览查询
    # 策略：文档提升 + LLM 重排（理解架构描述）
    QueryIntent.OVERVIEW: RetrievalStrategy(
        enable_path_index=False,
        enable_query_rewrite=False,
        enable_llm_rerank=True,
        boost_docs=True,
        max_chunks_per_path=3,
    ),
    
    # M (COMPOUND): 复合查询
    # 策略：查询改写 + LLM 重排（处理多条件）
    QueryIntent.COMPOUND: RetrievalStrategy(
        enable_path_index=False,
        enable_query_rewrite=True,
        enable_llm_rerank=True,
        max_chunks_per_path=3,
    ),
}


def get_strategy(intent: QueryIntent) -> RetrievalStrategy:
    """获取意图对应的检索策略
    
    Args:
        intent: 查询意图
        
    Returns:
        检索策略配置
    """
    return STRATEGY_TABLE.get(intent, STRATEGY_TABLE[QueryIntent.FEATURE])
