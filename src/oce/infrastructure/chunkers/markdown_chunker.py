"""Markdown chunker built on langchain's structural splitter.

Fixed line windows damage documents in two measured ways (147 files / 609
chunks): 26% of chunks start inside a fenced code block, so they open with
context-free code, and 20% contain no heading at all, so a hit gives no clue
what the passage is about.

``ExperimentalMarkdownSyntaxTextSplitter`` already solves the hard part: it
splits on heading hierarchy and treats fenced blocks as single units, verified
here to not mistake ``# comment`` inside a fence for a heading.

What it does not provide is line numbers, and ``Chunk`` requires text that
matches its reported span because the formatter prints ``start_line + offset``
per line. So its sections are used only to decide boundaries; the text is then
cut from the source lines. Every section it returns was verified to be a
verbatim substring of the source, which is what makes that lookup sound.
"""

from __future__ import annotations

from loguru import logger

from oce.domain.chunk.recursive_chunker import is_meaningful
from oce.domain.chunk.protocols import Chunker
from oce.domain.chunk.spans import cap_span, trim_trailing_blank_lines
from oce.domain.chunk.types import Chunk

DEFAULT_MAX_CHUNK_CHARS = 6_000
# Sections below this size are merged forward. Splitting on every heading level
# otherwise shatters reference docs into one- and two-line fragments that carry
# a title but no answer.
DEFAULT_MIN_CHUNK_CHARS = 700

_HEADERS_TO_SPLIT_ON = [
    ("#", "h1"),
    ("##", "h2"),
    ("###", "h3"),
]


class MarkdownChunker:
    """Structure-aware markdown chunker (implements chunk.protocols.Chunker)."""

    languages = frozenset({"markdown"})

    def __init__(
        self,
        *,
        fallback: Chunker,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
        min_chunk_chars: int = DEFAULT_MIN_CHUNK_CHARS,
    ):
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars 必须 > 0")
        if min_chunk_chars < 0 or min_chunk_chars >= max_chunk_chars:
            raise ValueError("min_chunk_chars 必须 ∈ [0, max_chunk_chars)")
        self.fallback = fallback
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, content: str, path: str) -> list[Chunk]:
        if not is_meaningful(content):
            return []
        lines = content.splitlines()
        if not lines:
            return []
        try:
            spans = self._section_spans(content, lines)
        except Exception as exc:
            logger.warning(
                "markdown 结构切块失败，退回行窗口: {}: {}",
                path,
                exc,
            )
            return self.fallback.chunk(content, path)
        if not spans:
            return self.fallback.chunk(content, path)
        return self._emit(spans, lines, path)

    def _section_spans(
        self,
        content: str,
        lines: list[str],
    ) -> list[tuple[int, int]]:
        """Map the splitter's sections onto 1-based inclusive line ranges."""
        starts = self._section_start_lines(content, lines)
        if not starts:
            return []
        spans = self._tile(self._drop_code_starts(starts, lines), len(lines))
        return self._merge_short(spans, lines)

    def _merge_short(
        self,
        spans: list[tuple[int, int]],
        lines: list[str],
    ) -> list[tuple[int, int]]:
        """Fold undersized sections into the following one.

        Reference docs nest headings densely, so honouring every boundary yields
        chunks holding a title and a single sentence. Those match a heading
        lexically but cannot answer anything, and they dilute the index. Merging
        forward keeps the outer heading at the top of the combined span.
        """
        if self.min_chunk_chars == 0:
            return spans
        merged: list[tuple[int, int]] = []
        pending_start: int | None = None
        for start, end in spans:
            current_start = pending_start if pending_start is not None else start
            size = sum(len(lines[row]) + 1 for row in range(current_start - 1, end))
            if size < self.min_chunk_chars:
                pending_start = current_start
                continue
            merged.append((current_start, end))
            pending_start = None
        if pending_start is not None:
            if merged:
                merged[-1] = (merged[-1][0], spans[-1][1])
            else:
                merged.append((pending_start, spans[-1][1]))
        return merged

    @staticmethod
    def _drop_code_starts(starts: list[int], lines: list[str]) -> list[int]:
        """Discard boundaries that would open a chunk on a fence marker.

        The splitter reports a fenced block as its own section. Cutting there
        would produce a chunk of bare code with no heading and no surrounding
        prose, which is exactly the failure mode this chunker exists to remove,
        so the block stays attached to the section that introduces it.
        """
        return [
            start
            for start in starts
            if not lines[start - 1].lstrip().startswith(("```", "~~~"))
        ]

    def _section_start_lines(self, content: str, lines: list[str]) -> list[int]:
        """Locate the first line of each section, heading included.

        The splitter reports a section's body, not its heading, so each located
        position is walked back over the heading line and any blank lines above
        it. Without that the boundary would fall after the heading and every
        chunk would end on the next section's title instead of opening with its
        own.

        The search advances a cursor so repeated text, such as two identical
        command blocks, does not collapse onto the same location.
        """
        from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter

        splitter = ExperimentalMarkdownSyntaxTextSplitter(
            headers_to_split_on=_HEADERS_TO_SPLIT_ON,
        )
        offsets = self._line_offsets(lines)
        starts: list[int] = []
        cursor = 0
        for section in splitter.split_text(content):
            text = section.page_content.strip()
            if not text:
                continue
            found = content.find(text, cursor)
            if found < 0:
                raise ValueError("section text is not a substring of the source")
            cursor = found + len(text)
            starts.append(
                self._claim_heading(lines, self._line_of(offsets, found))
            )
        return starts

    @staticmethod
    def _claim_heading(lines: list[str], body_line: int) -> int:
        """Walk back from a section body to the heading that introduces it."""
        candidate = body_line - 1
        while candidate >= 1 and not lines[candidate - 1].strip():
            candidate -= 1
        if candidate >= 1 and lines[candidate - 1].lstrip().startswith("#"):
            return candidate
        return body_line

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
        """Turn section start lines into contiguous, non-overlapping spans.

        Each section runs until the line before the next section starts, so a
        heading opens the chunk it belongs to. Any preamble above the first
        heading is folded into the first span, which keeps every line covered
        exactly once.
        """
        ordered = sorted({start for start in starts if 1 <= start <= total_lines})
        if not ordered:
            return []
        spans: list[tuple[int, int]] = []
        for index, start in enumerate(ordered):
            if index == 0:
                start = 1
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
                self.max_chunk_chars,
            ):
                if not text.strip():
                    continue
                chunks.append(
                    Chunk(
                        content_hash=Chunk.compute_hash(text),
                        path=path,
                        content=text,
                        start_line=span_start,
                        end_line=span_end,
                        chunk_type="markdown",
                    )
                )
        return chunks
