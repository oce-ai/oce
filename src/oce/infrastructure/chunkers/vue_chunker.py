"""Structure-aware chunking for Vue and Svelte single-file components."""

from __future__ import annotations

import os
import re
from bisect import bisect_right

from oce.domain.chunk.recursive_chunker import is_meaningful
from oce.domain.chunk.protocols import Chunker
from oce.domain.chunk.spans import cap_span, trim_trailing_blank_lines
from oce.domain.chunk.types import Chunk

DEFAULT_MAX_CHUNK_CHARS = 6_000
_SECTION_TAG = re.compile(
    r"<\s*(?P<closing>/)?\s*(?P<tag>template|script|style)\b[^>]*>",
    re.IGNORECASE,
)

Section = tuple[str, int, int]


class VueChunker:
    """Chunk Vue and Svelte components without separating tags from content."""

    languages = frozenset({"vue", "svelte"})

    def __init__(
        self,
        *,
        fallback: Chunker,
        max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    ):
        if max_chunk_chars <= 0:
            raise ValueError("max_chunk_chars 必须 > 0")
        self.fallback = fallback
        self.max_chunk_chars = max_chunk_chars

    def chunk(self, content: str, path: str) -> list[Chunk]:
        if not is_meaningful(content):
            return []
        lines = content.splitlines()
        if not lines:
            return []

        language = self._language(path)
        try:
            sections = self._locate_sections(content)
        except ValueError:
            return self.fallback.chunk(content, path)
        if language == "svelte":
            sections = [section for section in sections if section[0] in {"script", "style"}]
            sections.extend(self._locate_svelte_markup(lines, sections))
        if not sections:
            return self.fallback.chunk(content, path)
        chunks = self._emit(sections, lines, path, language)
        return chunks or self.fallback.chunk(content, path)

    @staticmethod
    def _language(path: str) -> str:
        return "svelte" if os.path.splitext(path.lower())[1] == ".svelte" else "vue"

    @staticmethod
    def _locate_sections(content: str) -> list[Section]:
        """Locate complete top-level SFC blocks as 1-based inclusive lines."""
        newline_offsets = [index for index, char in enumerate(content) if char == "\n"]
        sections: list[Section] = []
        active_tag: str | None = None
        active_start = 0
        template_depth = 0

        for match in _SECTION_TAG.finditer(content):
            tag = match.group("tag").lower()
            closing = match.group("closing") is not None
            if active_tag is None:
                if closing:
                    continue
                active_tag = tag
                active_start = match.start()
                template_depth = 1
                continue
            if tag != active_tag:
                continue
            if active_tag == "template" and not closing:
                template_depth += 1
                continue
            if not closing:
                continue
            template_depth -= 1
            if template_depth > 0:
                continue

            start_line = bisect_right(newline_offsets, active_start) + 1
            end_line = bisect_right(newline_offsets, match.end() - 1) + 1
            sections.append((active_tag, start_line, end_line))
            active_tag = None

        if active_tag is not None:
            raise ValueError(f"unclosed <{active_tag}> section")
        return sections

    @staticmethod
    def _locate_svelte_markup(
        lines: list[str],
        sections: list[Section],
    ) -> list[Section]:
        """Treat meaningful root content outside script/style as Svelte markup."""
        excluded: set[int] = set()
        for tag, start, end in sections:
            if tag in {"script", "style"}:
                excluded.update(range(start, end + 1))

        markup: list[Section] = []
        start: int | None = None
        for line_no, line in enumerate(lines, start=1):
            available = line_no not in excluded and bool(line.strip())
            if available and start is None:
                start = line_no
            if start is not None and (line_no in excluded or line_no == len(lines)):
                end = line_no - 1 if line_no in excluded else line_no
                while end >= start and not lines[end - 1].strip():
                    end -= 1
                if end >= start:
                    markup.append(("markup", start, end))
                start = None
        return markup

    def _emit(
        self,
        sections: list[Section],
        lines: list[str],
        path: str,
        language: str,
    ) -> list[Chunk]:
        styles = [section for section in sections if section[0] == "style"]
        primary = [section for section in sections if section[0] != "style"]
        groups = self._primary_groups(primary, styles, lines, language)
        groups.extend((start, end, "style") for _, start, end in styles)
        groups.sort(key=lambda group: group[0])

        chunks: list[Chunk] = []
        for start, end, section_type in groups:
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
                        chunk_type=f"{language}:{section_type}",
                    )
                )
        return chunks

    def _primary_groups(
        self,
        primary: list[Section],
        styles: list[Section],
        lines: list[str],
        language: str,
    ) -> list[tuple[int, int, str]]:
        if not primary:
            return []
        start = min(section[1] for section in primary)
        end = max(section[2] for section in primary)
        crosses_style = any(style_start <= end and style_end >= start for _, style_start, style_end in styles)
        tags = {section[0] for section in primary}
        if len(tags) == 1:
            combined_type = next(iter(tags))
        elif language == "vue":
            combined_type = "template+script"
        else:
            combined_type = "markup+script"
        combined_chars = len("\n".join(lines[start - 1 : end]))
        if not crosses_style and combined_chars <= self.max_chunk_chars:
            return [(start, end, combined_type)]
        return [(section_start, section_end, tag) for tag, section_start, section_end in primary]
