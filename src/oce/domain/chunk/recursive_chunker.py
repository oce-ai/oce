"""RecursiveChunker — 基于 LangChain RecursiveCharacterTextSplitter 的智能兜底切块器。

用于：
1. 无法识别语言的文件（无扩展名、未知格式）
2. 各专用 chunker 的 fallback（当 AST 解析失败时）

相比 FixedChunker 的优势：
- 递归尝试分隔符：优先在段落边界（\n\n）切分，其次行边界（\n），最后空格和字符
- 语言感知：支持 Python/JS/Go/Rust 等语言的特定分隔符

The splitter decides where boundaries fall; the text of a chunk is always cut
from the source lines, the same contract ``CastChunker`` and ``MarkdownChunker``
follow. Its own output cannot serve as chunk text: it strips whitespace at every
separator, so the returned string no longer matches the lines it came from and
the line number derived from it points at the wrong place. Measured on the
OpenClaw tree, 13.3% of the chunks produced that way reported a range whose
source text differed from the chunk itself.
"""
from __future__ import annotations

from loguru import logger

from oce.domain.chunk.lang import detect_language
from oce.domain.chunk.protocols import Chunker
from oce.domain.chunk.spans import cap_span, trim_trailing_blank_lines
from oce.domain.chunk.types import Chunk

DEFAULT_MAX_CHUNK_CHARS = 6_000
DEFAULT_CHUNK_OVERLAP = 200

# Hard ceiling on a chunk's text, independent of ``chunk_size``. The two differ
# in kind: ``chunk_size`` is how large a chunk should be, this is how large one
# may be before the embedding client re-splits it and the vector store truncates
# its text field. A single line longer than this is dropped, so tying the
# ceiling to ``chunk_size`` would discard ordinary long lines whenever a caller
# asked for small chunks.
MAX_SPAN_CHARS = 6_000


def is_meaningful(text: str | bytes) -> bool:
    """是否含有效信息（至少一个字母/数字）。"""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    return any(ch.isalnum() for ch in text)


class RecursiveChunker:
    """基于 LangChain RecursiveCharacterTextSplitter 的通用 Chunker。"""

    def __init__(
        self,
        chunk_size: int = DEFAULT_MAX_CHUNK_CHARS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """``chunk_overlap`` 只影响 splitter 在何处切分，不产生块间重叠。

        重复的文本对带行号的 chunk 是有害的：两个块声明同样的行，就会把同一段
        源码索引两遍，并占掉两个检索位。
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size 必须 > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须 ∈ [0, chunk_size)")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, content: str, path: str) -> list[Chunk]:
        # 处理 bytes 输入（从 staging 读取的内容）
        if isinstance(content, bytes):
            content = content.decode("utf-8", errors="ignore")

        if not is_meaningful(content):
            return []

        lines = content.splitlines()
        if not lines:
            return []

        splitter = self._create_splitter(detect_language(path))
        # create_documents 而不是 split_text：只有它会带上 start_index，而起始
        # 偏移是把切点映射回行号的唯一可靠依据。用 find() 反查文本会在 splitter
        # strip 掉空白后落到错误的行上。
        try:
            pieces = splitter.create_documents([content])
        except Exception as error:
            logger.debug(f"RecursiveCharacterTextSplitter failed for {path}: {error}")
            return []
        starts = self._start_lines(content, lines, pieces)
        if not starts:
            return []
        return self._emit(self._tile(starts, len(lines)), lines, path)

    def _create_splitter(self, language: str | None) -> RecursiveCharacterTextSplitter:
        """创建适合语言的 splitter。"""
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_text_splitters.base import Language

        if language:
            try:
                # LangChain 的语言映射
                lang_map = {
                    "python": Language.PYTHON,
                    "javascript": Language.JS,
                    "typescript": Language.TS,
                    "java": Language.JAVA,
                    "cpp": Language.CPP,
                    "go": Language.GO,
                    "rust": Language.RUST,
                    "markdown": Language.MARKDOWN,
                    "html": Language.HTML,
                }
                lang_enum = lang_map.get(language.lower())
                if lang_enum:
                    return RecursiveCharacterTextSplitter.from_language(
                        language=lang_enum,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                        add_start_index=True,
                    )
            except Exception as e:
                logger.debug(f"Failed to use language-specific splitter for {language}: {e}")

        # 默认通用分隔符
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True,
        )

    def _start_lines(
        self,
        content: str,
        lines: list[str],
        pieces: list,
    ) -> list[int]:
        """Map each piece's start offset onto the 1-based line that owns it.

        A boundary landing mid-line is pulled back to that line's start: a chunk
        may not begin halfway through a line, or its text would no longer be the
        lines it claims. Splitting mid-line only happens once the splitter has
        exhausted its line-based separators, on input with no line structure
        left to respect.
        """
        offsets = self._line_offsets(lines)
        starts: list[int] = []
        for piece in pieces:
            position = piece.metadata.get("start_index")
            # start_index 来自 splitter 内部的 str.find，找不到时是 -1。缺失或
            # 找不到都意味着无从确定行号，此时宁可少切一刀，也不能报错的位置。
            if position is None or position < 0:
                continue
            starts.append(self._line_of(offsets, min(position, len(content))))
        return starts or [1]

    @staticmethod
    def _line_offsets(lines: list[str]) -> list[int]:
        """Character offset of each line's first character."""
        offsets: list[int] = []
        position = 0
        for line in lines:
            offsets.append(position)
            position += len(line) + 1
        return offsets

    @staticmethod
    def _line_of(offsets: list[int], position: int) -> int:
        """Binary-search the 1-based line owning a character offset."""
        low, high = 0, len(offsets) - 1
        while low < high:
            mid = (low + high + 1) // 2
            if offsets[mid] <= position:
                low = mid
            else:
                high = mid - 1
        return low + 1

    @staticmethod
    def _tile(starts: list[int], total_lines: int) -> list[tuple[int, int]]:
        """Turn start lines into contiguous, non-overlapping line ranges.

        Overlap is dropped here. The splitter is configured to repeat text
        between neighbours, which is meaningless once chunks carry line numbers:
        two chunks claiming the same lines index the same source twice and take
        two retrieval slots to say one thing.
        """
        ordered = sorted({start for start in starts if 1 <= start <= total_lines})
        if not ordered or ordered[0] != 1:
            ordered.insert(0, 1)
        spans: list[tuple[int, int]] = []
        for index, start in enumerate(ordered):
            end = ordered[index + 1] - 1 if index + 1 < len(ordered) else total_lines
            if end >= start:
                spans.append((start, end))
        return spans

    def _emit(
        self,
        spans: list[tuple[int, int]],
        lines: list[str],
        path: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for start, end in spans:
            trimmed = trim_trailing_blank_lines(lines, start, end)
            for span_start, span_end, text in cap_span(
                lines,
                start,
                trimmed,
                max(self.chunk_size, MAX_SPAN_CHARS),
            ):
                if not is_meaningful(text):
                    continue
                chunks.append(
                    Chunk(
                        content_hash=Chunk.compute_hash(text),
                        path=path,
                        content=text,
                        start_line=span_start,
                        end_line=span_end,
                        chunk_type="recursive",
                    )
                )
        return chunks
