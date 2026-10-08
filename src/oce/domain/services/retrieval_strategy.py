"""意图驱动的检索策略决策表。

这张表同时是标签体系的**证明**：标签存在的唯一理由是它能导出一条与其他
标签不同的检索行为。`tests/unit/domain/test_retrieval_strategy.py` 断言
任意两个意图的策略配置不完全相同——若两行配置一样，说明这两个标签在检索
层没有区别，应该合并而不是并存。

重构前 `FEATURE` 与 `COMPOUND` 的配置一字不差，且 M 真正需要的按子句多路
召回（`split_independent_clauses`）根本没进这张表，只在检索流程里以检测器
形式隐式存在。现在 `split_clauses` 是显式字段。
"""

from __future__ import annotations

from dataclasses import dataclass

from oce.domain.services.intent.taxonomy import QueryIntent


@dataclass(frozen=True, slots=True)
class RetrievalStrategy:
    """一个意图对应的检索策略配置。"""

    enable_path_index: bool = False       # 启用路径索引
    enable_query_rewrite: bool = False    # 启用查询改写
    enable_llm_rerank: bool = False       # 启用 LLM 重排
    boost_definitions: bool = False       # 提升定义位置权重
    boost_docs: bool = False              # 提升文档权重
    max_chunks_per_path: int = 3          # 每个文件最多返回块数
    #: 按独立子句拆分为多路召回。只有 M 需要，是 M 区别于 F 的实质所在。
    split_clauses: bool = False
    #: 广度优先：引用点查询要跨文件铺开，而非集中在少数权威位置。
    prefer_breadth: bool = False
    enable_multi_hop: bool = False        # 多跳检索（调用链），尚未实现
    enable_reference_graph: bool = False  # 引用图（依赖分析），尚未实现


#: 决策表：意图 -> 检索策略。任意两行必须不同（有测试断言）。
STRATEGY_TABLE: dict[QueryIntent, RetrievalStrategy] = {
    # S：符号定义。定义位置加权，块数少而精；路径语义会把同名引用顶到定义前面。
    QueryIntent.SYMBOL: RetrievalStrategy(
        enable_query_rewrite=True,
        enable_llm_rerank=True,
        boost_definitions=True,
        max_chunks_per_path=2,
    ),
    # C：调用链。保留原查询的方向与边界信息，不改写；交给 LLM 判断调用关系。
    QueryIntent.CALL_CHAIN: RetrievalStrategy(
        enable_llm_rerank=True,
        max_chunks_per_path=3,
        enable_multi_hop=False,
    ),
    # R：API 契约。要签名与调用示例，位置权威且少量；不需要广度。
    QueryIntent.REFERENCE: RetrievalStrategy(
        enable_query_rewrite=True,
        enable_llm_rerank=True,
        max_chunks_per_path=2,
    ),
    # U：引用点。要跨文件铺开所有调用处，不要定义加权，块数放宽。
    # 这三点合起来构成 U 区别于 S 与 R 的实质。
    QueryIntent.USAGE: RetrievalStrategy(
        enable_query_rewrite=True,
        enable_llm_rerank=False,
        boost_definitions=False,
        max_chunks_per_path=5,
        prefer_breadth=True,
        enable_reference_graph=False,
    ),
    # P：文件路径。路径索引负责召回，改写补中英文差异，LLM 定最终顺序。
    QueryIntent.PATH: RetrievalStrategy(
        enable_path_index=True,
        enable_query_rewrite=True,
        enable_llm_rerank=True,
        max_chunks_per_path=2,
    ),
    # F：功能实现。跨中英文术语召回，再由正文相关性确定实现文件。
    QueryIntent.FEATURE: RetrievalStrategy(
        enable_query_rewrite=True,
        enable_llm_rerank=True,
        max_chunks_per_path=3,
    ),
    # O：架构概览。提升文档权重，不改写（架构措辞改写后容易失真）。
    QueryIntent.OVERVIEW: RetrievalStrategy(
        enable_llm_rerank=True,
        boost_docs=True,
        max_chunks_per_path=3,
    ),
    # M：复合查询。按独立子句拆成多路召回，每路独立规划/召回再融合。
    QueryIntent.COMPOUND: RetrievalStrategy(
        enable_query_rewrite=True,
        enable_llm_rerank=True,
        max_chunks_per_path=3,
        split_clauses=True,
    ),
}


def get_strategy(intent: QueryIntent) -> RetrievalStrategy:
    """取意图对应的检索策略；未知意图退回 FEATURE 的保守配置。"""
    return STRATEGY_TABLE.get(intent, STRATEGY_TABLE[QueryIntent.FEATURE])
