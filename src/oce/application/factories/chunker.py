"""Chunker 装配:把 infrastructure 的各语言 chunker 组装进 router。"""

from oce.domain.chunk import LanguageChunkerRouter
from oce.domain.chunk.recursive_chunker import RecursiveChunker
from oce.infrastructure.astchunk.cast_chunker import CastChunker
from oce.infrastructure.chunkers.jsp_chunker import JspChunker
from oce.infrastructure.chunkers.markdown_chunker import MarkdownChunker
from oce.infrastructure.chunkers.vue_chunker import VueChunker


def build_chunker() -> LanguageChunkerRouter:
    """构建 Chunker router，使用 RecursiveChunker 作为统一 fallback。

    架构说明：
    - RecursiveChunker: 基于 LangChain，智能递归分隔，支持语言特定规则
    - CastChunker: AST 语义切块，内部自带 RecursiveCharacterTextSplitter fallback
    - 各专用 chunker (Markdown/JSP/Vue): 针对特定格式优化
    - FixedChunker 已废弃，完全由 RecursiveChunker 替代
    """
    recursive_chunker = RecursiveChunker(chunk_size=6000, chunk_overlap=200)

    return LanguageChunkerRouter(
        fallback=recursive_chunker,
        language_chunkers=(
            CastChunker(
                max_chunk_size=1500,
                chunk_overlap=0,
                fallback=recursive_chunker,
            ),
            MarkdownChunker(fallback=recursive_chunker),
            JspChunker(fallback=recursive_chunker),
            VueChunker(fallback=recursive_chunker),
        ),
    )
