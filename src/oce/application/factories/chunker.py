"""Chunker 装配:把 infrastructure 的各语言 chunker 组装进 router。

尺寸旋钮来自 ``ChunkingSettings``（env 前缀 ``CHUNK_``），不再硬编码——chunk_size 是检索
质量最核心的 L1 旋钮，提配置后才能被评测 profile 驱动并在 RunRecord 里归因（见 bench）。
默认值与提配置前逐字一致，故本工厂的产出在未覆盖配置时与历史完全等价。
"""

from oce.domain.chunk import LanguageChunkerRouter
from oce.domain.chunk.recursive_chunker import RecursiveChunker
from oce.infrastructure.astchunk.cast_chunker import CastChunker
from oce.infrastructure.chunkers.jsp_chunker import JspChunker
from oce.infrastructure.chunkers.markdown_chunker import MarkdownChunker
from oce.infrastructure.chunkers.vue_chunker import VueChunker
from oce.shared.config.settings import ChunkingSettings


def build_chunker(chunking: ChunkingSettings) -> LanguageChunkerRouter:
    """构建 Chunker router，使用 RecursiveChunker 作为统一 fallback。

    架构说明：
    - RecursiveChunker: 基于 LangChain，智能递归分隔，支持语言特定规则
    - CastChunker: AST 语义切块，内部自带 RecursiveCharacterTextSplitter fallback
    - 各专用 chunker (Markdown/JSP/Vue): 针对特定格式优化
    - FixedChunker 已废弃，完全由 RecursiveChunker 替代
    """
    recursive_chunker = RecursiveChunker(
        chunk_size=chunking.recursive_size,
        chunk_overlap=chunking.recursive_overlap,
    )

    return LanguageChunkerRouter(
        fallback=recursive_chunker,
        language_chunkers=(
            CastChunker(
                max_chunk_size=chunking.ast_max_size,
                chunk_overlap=chunking.ast_overlap,
                fallback=recursive_chunker,
            ),
            MarkdownChunker(fallback=recursive_chunker),
            JspChunker(fallback=recursive_chunker),
            VueChunker(fallback=recursive_chunker),
        ),
    )
