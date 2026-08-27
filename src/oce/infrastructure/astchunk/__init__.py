"""AST-aware source chunking infrastructure."""

from .astchunk import ASTChunk
from .astchunk_builder import ASTChunkBuilder, LANGUAGE_MAP, get_supported_languages
from .astnode import ASTNode
from .preprocessing import (
    ByteRange,
    IntRange,
    get_largest_node_in_brange,
    get_nodes_in_brange,
    get_nws_count,
    get_nws_count_direct,
    preprocess_nws_count,
)

__version__ = "0.1.0"
