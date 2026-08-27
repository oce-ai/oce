"""Structure-aware chunking for JSP pages, fragments, and tag files."""

from __future__ import annotations

import re
from collections.abc import Iterable

from loguru import logger
from tree_sitter_language_pack import get_parser

from oce.domain.chunk.recursive_chunker import is_meaningful
from oce.domain.chunk.protocols import Chunker
from oce.domain.chunk.spans import cap_span, trim_trailing_blank_lines
from oce.domain.chunk.types import Chunk
from oce.infrastructure.astchunk.compat import CompatNode, compat_parse

DEFAULT_MAX_CHUNK_CHARS = 6_000
_JSP_BLOCK = re.compile(r"<%.*?%>", re.DOTALL)
_JSP_XML_CODE = re.compile(
    r"(?P<open><jsp:(?P<kind>scriptlet|expression|declaration)\b[^>]*>)"
    r"(?P<body>.*?)"
    r"(?P<close></jsp:(?P=kind)\s*>)",
    re.DOTALL | re.IGNORECASE,
)
_CONTENT_NODE_TYPES = frozenset({"element", "script_element", "style_element"})

Boundary = tuple[int, str]


class JspChunker:
    """Split JSP-family templates at top-level rendered-content boundaries."""

    languages = frozenset({"jsp"})

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
        self._parser = get_parser("html")

    def chunk(self, content: str, path: str) -> list[Chunk]:
        if not is_meaningful(content):
            return []
        lines = content.splitlines()
        if not lines:
            return []

        try:
            masked = self._mask_jsp_code(content)
            root = compat_parse(self._parser, masked).root_node
            boundaries = self._content_boundaries(root)
        except Exception as exc:
            logger.warning("JSP 结构切块失败，退回行窗口: {}: {}", path, exc)
            return self.fallback.chunk(content, path)
        if not boundaries:
            return self.fallback.chunk(content, path)
        return self._emit(boundaries, lines, path)

    @classmethod
    def _mask_jsp_code(cls, content: str) -> str:
        """Hide embedded Java while preserving every source line and column."""
        masked = _JSP_BLOCK.sub(lambda match: cls._blank(match.group(0)), content)

        def blank_xml_body(match: re.Match[str]) -> str:
            return (
                match.group("open")
                + cls._blank(match.group("body"))
                + match.group("close")
            )

        return _JSP_XML_CODE.sub(blank_xml_body, masked)

    @staticmethod
    def _blank(value: str) -> str:
        return "".join("\n" if char == "\n" else " " for char in value)

    def _content_boundaries(self, root: CompatNode) -> list[Boundary]:
        body = self._find_element(root, "body")
        if body is not None:
            boundaries = self._direct_content_children(body)
            if boundaries:
                return boundaries

        top_level = [
            child
            for child in root.children
            if child.type in _CONTENT_NODE_TYPES
        ]
        if len(top_level) == 1 and self._tag_name(top_level[0]) in {
            "html",
            "jsp:root",
        }:
            nested = self._direct_content_children(top_level[0])
            if len(nested) > 1:
                return nested
        return self._as_boundaries(top_level)

    def _find_element(self, node: CompatNode, tag_name: str) -> CompatNode | None:
        if node.type in _CONTENT_NODE_TYPES and self._tag_name(node) == tag_name:
            return node
        for child in node.children:
            found = self._find_element(child, tag_name)
            if found is not None:
                return found
        return None

    def _direct_content_children(self, node: CompatNode) -> list[Boundary]:
        return self._as_boundaries(
            child for child in node.children if child.type in _CONTENT_NODE_TYPES
        )

    def _as_boundaries(self, nodes: Iterable[CompatNode]) -> list[Boundary]:
        boundaries: list[Boundary] = []
        for node in nodes:
            line = node.start_point.row + 1
            tag = self._tag_name(node) or "section"
            if boundaries and boundaries[-1][0] == line:
                continue
            boundaries.append((line, tag))
        return boundaries

    @staticmethod
    def _tag_name(node: CompatNode) -> str | None:
        pending = list(node.children)
        while pending:
            current = pending.pop(0)
            if current.type == "tag_name":
                return current.text.decode("utf8", errors="replace").lower()
            if current.type in {"start_tag", "self_closing_tag"}:
                pending[0:0] = current.children
        return None

    def _emit(
        self,
        boundaries: list[Boundary],
        lines: list[str],
        path: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, (boundary_start, tag) in enumerate(boundaries):
            start = 1 if index == 0 else boundary_start
            end = (
                boundaries[index + 1][0] - 1
                if index + 1 < len(boundaries)
                else len(lines)
            )
            end = trim_trailing_blank_lines(lines, start, end)
            for span_start, span_end, text in cap_span(
                lines,
                start,
                end,
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
                        chunk_type=f"jsp:{tag}",
                    )
                )
        return chunks
