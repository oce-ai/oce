"""MarkdownChunker behaviour: fence safety, heading anchors, span alignment.

The point of structural splitting is that a document chunk is usable on its own:
it should not open in the middle of a fenced code block, and it should carry the
heading that introduces it. Line numbers must still match the text exactly,
since the formatter renders each line at ``start_line + offset``.
"""

from __future__ import annotations

import pytest

from oce.domain.chunk import RecursiveChunker, LanguageChunker
from oce.infrastructure.chunkers.markdown_chunker import MarkdownChunker

DOC = """# Guide

Intro paragraph.

## Install

Run the installer:

```bash
# this comment is not a heading
./install.sh --yes
```

## Configure

Set the token.

### Advanced

Tune the cache.
"""


def make_chunker(**kwargs) -> MarkdownChunker:
    return MarkdownChunker(fallback=RecursiveChunker(), **kwargs)


def assert_aligned(chunk, content: str) -> None:
    lines = content.splitlines()
    expected = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
    assert chunk.content == expected, (
        f"{chunk.path}#{chunk.start_line}-{chunk.end_line} text does not match its span"
    )


def fence_state_per_line(content: str) -> list[bool]:
    """Whether each line sits inside a fenced block, marker lines included."""
    inside: list[bool] = []
    open_marker: str | None = None
    for line in content.splitlines():
        stripped = line.lstrip()
        marker = next(
            (m for m in ("```", "~~~") if stripped.startswith(m)),
            None,
        )
        if marker is not None and open_marker is None:
            inside.append(True)
            open_marker = marker
        elif marker is not None and marker == open_marker:
            inside.append(True)
            open_marker = None
        else:
            inside.append(open_marker is not None)
    return inside


class TestMarkdownChunker:
    def test_declares_its_language_capability(self):
        chunker = make_chunker()

        assert isinstance(chunker, LanguageChunker)
        assert chunker.languages == frozenset({"markdown"})

    def test_chunks_stay_aligned_with_their_spans(self):
        chunks = make_chunker().chunk(DOC, "docs/guide.md")
        assert chunks
        for chunk in chunks:
            assert_aligned(chunk, DOC)

    def test_every_line_is_covered_exactly_once(self):
        chunks = make_chunker().chunk(DOC, "docs/guide.md")
        seen: dict[int, int] = {}
        for chunk in chunks:
            for line_no in range(chunk.start_line, chunk.end_line + 1):
                seen[line_no] = seen.get(line_no, 0) + 1
        meaningful = [
            index
            for index, line in enumerate(DOC.splitlines(), 1)
            if line.strip()
        ]
        assert all(seen.get(line_no) == 1 for line_no in meaningful)

    def test_no_chunk_starts_inside_a_fence(self):
        chunks = make_chunker().chunk(DOC, "docs/guide.md")
        inside = fence_state_per_line(DOC)
        for chunk in chunks:
            assert not inside[chunk.start_line - 1], (
                f"chunk at {chunk.start_line} starts inside a fenced block"
            )

    def test_fence_comment_is_not_treated_as_a_heading(self):
        chunks = make_chunker().chunk(DOC, "docs/guide.md")
        fenced = next(c for c in chunks if "./install.sh" in c.content)
        assert "# this comment is not a heading" in fenced.content
        assert "## Install" in fenced.content

    def test_sections_carry_their_heading(self):
        chunks = make_chunker().chunk(DOC, "docs/guide.md")
        with_heading = [
            chunk
            for chunk in chunks
            if any(
                line.lstrip().startswith("#")
                for line in chunk.content.splitlines()
            )
        ]
        assert len(with_heading) == len(chunks)

    def test_chunk_type_is_markdown(self):
        chunks = make_chunker().chunk(DOC, "docs/guide.md")
        assert {chunk.chunk_type for chunk in chunks} == {"markdown"}

    def test_character_budget_is_enforced(self):
        body = "\n".join(f"detail line {index}" for index in range(400))
        content = f"# Big\n\n{body}\n"
        chunks = make_chunker(max_chunk_chars=1_000).chunk(content, "docs/big.md")
        assert chunks
        assert all(len(chunk.content) <= 1_000 for chunk in chunks)
        for chunk in chunks:
            assert_aligned(chunk, content)

    def test_blank_document_yields_nothing(self):
        assert make_chunker().chunk("\n\n   \n", "docs/empty.md") == []

    def test_document_without_headings_still_chunks(self):
        content = "\n".join(f"plain line {index}" for index in range(30))
        chunks = make_chunker().chunk(content, "docs/plain.md")
        assert chunks
        for chunk in chunks:
            assert_aligned(chunk, content)

    def test_falls_back_when_structural_split_fails(self, monkeypatch):
        chunker = make_chunker()
        monkeypatch.setattr(
            chunker,
            "_section_spans",
            lambda *_: (_ for _ in ()).throw(RuntimeError("splitter unavailable")),
        )
        chunks = chunker.chunk(DOC, "docs/guide.md")
        assert chunks
        assert all(chunk.chunk_type == "recursive" for chunk in chunks)

    def test_rejects_non_positive_budget(self):
        with pytest.raises(ValueError):
            make_chunker(max_chunk_chars=0)
