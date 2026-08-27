"""代码切块协议、值对象和纯领域实现。"""

from oce.domain.chunk.protocols import Chunker, LanguageChunker
from oce.domain.chunk.recursive_chunker import RecursiveChunker, is_meaningful
from oce.domain.chunk.router import LanguageChunkerRouter
from oce.domain.chunk.types import Chunk, ChunkRef, LocatedChunk
