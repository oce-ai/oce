"""cAST chunker adapter with a configurable fallback.

The AST decides where chunk boundaries fall; the text of a chunk is always cut
from the source lines. astchunk rebuilds text from node coordinates and pads
gaps with spaces, so its output is not byte-identical to the file and cannot be
trusted to line up with the line numbers the formatter prints.
"""

from __future__ import annotations

from loguru import logger

from oce.domain.chunk.lang import SUPPORTED_LANGUAGES, detect_language
from oce.domain.chunk.protocols import Chunker
from oce.domain.chunk.spans import cap_span, trim_trailing_blank_lines
from oce.domain.chunk.types import Chunk
from oce.infrastructure.astchunk.astchunk_builder import ASTChunkBuilder, LANGUAGE_MAP


# Keeps chunk text inside the embedding client's per-input window and the vector
# store's text field. Above it the client re-splits and pools the pieces and the
# store truncates, so the indexed text stops matching the reported line range.
DEFAULT_MAX_CHUNK_CHARS = 6_000

# Non-whitespace characters per character of source, at the low end. Measured
# over the supported grammars on the OpenClaw tree: Swift sits lowest at 0.67,
# Dockerfile highest at 0.88. Converting the character budget with the floor
# means the derived non-whitespace budget holds for every language, at the cost
# of leaving some headroom unused in the denser ones.
_MIN_NWS_DENSITY = 0.67

# Ranges holding less than this are folded into a neighbour. astchunk measures
# windows in non-whitespace characters and splits on column offsets, so a
# declaration whose body lands in the next window comes back as a line or two
# holding just a signature or a closing brace. Indexed on its own such a chunk
# matches a name but answers nothing, and it takes a retrieval slot from the
# body that does.
DEFAULT_MIN_CHUNK_CHARS = 300


class CastChunker:
    """AST implementation for programming languages with semantic parsers.

    架构说明：
    - 有语言识别 → 交给 ASTChunkBuilder（内部自动 fallback 到 RecursiveCharacterTextSplitter）
    - 无语言识别 → 使用外层 RecursiveChunker fallback
    - 不再在外层捕获异常，信任 ASTChunkBuilder 的处理

    排除有专用 chunker 的语言（markdown、jsp、vue、svelte）
    """

    # 排除有专用 chunker 的语言
    _EXCLUDED_LANGUAGES = frozenset({"markdown", "jsp", "vue", "svelte"})
    languages = frozenset(
        SUPPORTED_LANGUAGES.intersection(LANGUAGE_MAP) - _EXCLUDED_LANGUAGES
    )

    def __init__(
        self,
        *,
        max_chunk_size: int,
        chunk_overlap: int,
        fallback: Chunker | None = None,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    ):
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size 必须 > 0")
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars 必须 > 0")
        if min_chunk_chars < 0 or min_chunk_chars >= max_chunk_chars:
            raise ValueError("min_chunk_chars 必须 ∈ [0, max_chunk_chars)")
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.fallback = fallback
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self._builders: dict[str, ASTChunkBuilder] = {}

    def chunk(self, content: str, path: str) -> list[Chunk]:
        if content == "":
            return []
        language = detect_language(path)
        if language is None:
            # 完全无法识别的文件（无扩展名且非特殊文件名）
            return self.fallback.chunk(content, path)
        # ASTChunkBuilder 内部会自动判断语言是否支持以及是否需要 fallback
        return self._chunk_ast(content, path, language)

    def _chunk_ast(self, content: str, path: str, language: str) -> list[Chunk]:
        raw_chunks = self._get_builder(language).chunkify(
            content,
            repo_level_metadata={"filepath": path},
            chunk_overlap=self.chunk_overlap,
        )
        lines = content.splitlines()
        line_count = len(lines)
        ranges = [
            (*self._resolve_range(item.get("metadata", {}), lines, line_count),
             item.get("metadata", {}).get("type", "ast"))
            for item in raw_chunks
        ]
        chunks: list[Chunk] = []
        for start, end, chunk_type in self._merge_small(ranges, lines):
            for span_start, span_end, text in cap_span(
                lines,
                start,
                end,
                self.max_chunk_chars,
            ):
                chunks.append(
                    Chunk(
                        content_hash=Chunk.compute_hash(text),
                        path=path,
                        content=text,
                        start_line=span_start,
                        end_line=span_end,
                        chunk_type=chunk_type,
                    )
                )
        if chunks:
            return chunks
        # 解析成功但所有行都超出字符预算：文件是压缩包或单行生成产物。
        # 回退到 RecursiveChunker 只会把同样的内容按字符切回来，所以不产出。
        if raw_chunks:
            return []
        return self.fallback.chunk(content, path)

    def _merge_small(
        self,
        ranges: list[tuple[int, int, str]],
        lines: list[str],
    ) -> list[tuple[int, int, str]]:
        """Fold undersized ranges into a neighbour, keeping coverage in order.

        astchunk measures a window in non-whitespace characters and cuts on
        column offsets, so one source construct can come back as two windows
        starting on the same line: the first holds only what precedes the split
        column, the second the rest. Both then resolve to line ranges where the
        first is a prefix of the second, and emitting it separately indexes a
        bare signature.

        A range is absorbed when it is already covered by what has been kept, or
        when its text is below the floor. Absorption extends the previous kept
        range instead of dropping lines, so the union of the output still covers
        the union of the input.
        """
        if self.min_chunk_chars == 0:
            return ranges
        merged: list[tuple[int, int, str]] = []
        for start, end, chunk_type in sorted(ranges):
            if merged:
                prev_start, prev_end, _ = merged[-1]
                # 已被前一个区间覆盖：重复起点或完全内含，不产出新块。
                if end <= prev_end:
                    continue
                # 前一个区间还没达到下限，把当前区间并进去补足它。类型取自
                # 被吞并的区间：过小的那一半通常只是签名前缀，描述这段代码的
                # 是带上了主体的这一个。
                if self._span_chars(lines, prev_start, prev_end) < self.min_chunk_chars:
                    merged[-1] = (prev_start, end, chunk_type)
                    continue
            if (
                merged
                and self._span_chars(lines, start, end) < self.min_chunk_chars
            ):
                # 自身过小且前一个已达标：向前贴，避免留下孤立的收尾括号。
                prev_start, _, prev_type = merged[-1]
                merged[-1] = (prev_start, end, prev_type)
                continue
            merged.append((start, end, chunk_type))
        return merged

    @staticmethod
    def _span_chars(lines: list[str], start: int, end: int) -> int:
        """Character count of a 1-based inclusive line range, newlines included."""
        return sum(len(lines[row]) + 1 for row in range(start - 1, end)) - 1

    def _resolve_range(
        self,
        metadata: dict,
        lines: list[str],
        line_count: int,
    ) -> tuple[int, int]:
        """Convert astchunk's 0-based rows into a 1-based inclusive line range.

        astchunk reports the row of the node's last byte. When a node ends at
        column 0 it stopped at the line break, so that row belongs to the next
        chunk; otherwise the row is part of this one. Trailing blank lines are
        dropped because joined text would never reach them.
        """
        start_row = int(metadata.get("start_line_no", 0))
        end_row = int(metadata.get("end_line_no", start_row))
        end_column = metadata.get("end_column")
        start = start_row + 1
        if end_column is not None and int(end_column) == 0 and end_row > start_row:
            end = end_row
        else:
            end = end_row + 1
        if start < 1 or end < start or end > line_count:
            raise ValueError(
                f"astchunk 返回无效行号: {start}-{end}，文件共 {line_count} 行"
            )
        return start, trim_trailing_blank_lines(lines, start, end)

    def _get_builder(self, language: str) -> ASTChunkBuilder:
        if language not in self._builders:
            self._builders[language] = ASTChunkBuilder(
                max_chunk_size=self.max_chunk_size,
                language=language,
                metadata_template="default",
                intact_node_size=self._intact_node_size(),
            )
        return self._builders[language]

    def _intact_node_size(self) -> int:
        """Largest declaration, in non-whitespace characters, kept whole.

        The two budgets measure different things: the builder counts
        non-whitespace characters, ``cap_span`` counts every character. Deriving
        one from the other keeps them from working against each other — a
        declaration held together by the builder only to be cut by ``cap_span``
        ends up in the same broken halves the intact rule exists to avoid.

        ``_MIN_NWS_DENSITY`` is the floor measured across the supported
        languages (Swift indents most heavily, at 0.67); using the floor rather
        than a per-language figure means the bound holds before any of the file
        has been read.

        The window size is the lower bound because a ceiling below it would
        reject nodes the builder is willing to place whole anyway, which is not
        this rule's business.
        """
        derived = int(self.max_chunk_chars * _MIN_NWS_DENSITY)
        return max(self.max_chunk_size, derived)
