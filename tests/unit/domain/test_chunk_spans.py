"""Line-span alignment invariants for both chunkers.

The formatter renders each chunk line at ``start_line + offset``, so a chunk
whose text does not match its declared line range reports wrong line numbers to
callers. These tests pin that invariant plus the character budget that keeps
chunk text from being re-split by the embedder or truncated by the vector store.
"""

from __future__ import annotations

import pytest

from oce.domain.chunk import RecursiveChunker
from oce.domain.chunk.spans import cap_span, trim_trailing_blank_lines
from oce.infrastructure.astchunk.cast_chunker import CastChunker


def assert_aligned(chunk, content: str) -> None:
    """A chunk's text must equal the source lines it claims."""
    lines = content.splitlines()
    expected = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
    assert chunk.content == expected, (
        f"{chunk.path}#{chunk.start_line}-{chunk.end_line} text does not match its span"
    )


class TestSpans:
    def test_cap_span_keeps_line_numbers_contiguous(self):
        lines = [f"line{index}" for index in range(1, 11)]
        spans = cap_span(lines, 1, 10, 20)
        assert spans[0][0] == 1
        assert spans[-1][1] == 10
        for previous, current in zip(spans, spans[1:]):
            assert current[0] == previous[1] + 1

    def test_cap_span_text_matches_claimed_lines(self):
        lines = [f"value-{index}" for index in range(1, 21)]
        for start, end, text in cap_span(lines, 1, 20, 30):
            assert text == "\n".join(lines[start - 1 : end])

    def test_overlong_single_line_is_dropped(self):
        """切片会让每一片都错报自己的行范围，所以整行不产出。"""
        assert cap_span(["x" * 250], 1, 1, 100) == []

    def test_overlong_line_does_not_merge_the_lines_around_it(self):
        lines = ["before", "y" * 250, "after"]
        spans = cap_span(lines, 1, 3, 100)
        assert [(start, end, text) for start, end, text in spans] == [
            (1, 1, "before"),
            (3, 3, "after"),
        ]

    def test_spans_stay_aligned_when_an_overlong_line_is_dropped(self):
        lines = ["alpha", "z" * 400, "beta", "gamma"]
        for start, end, text in cap_span(lines, 1, 4, 100):
            assert text == "\n".join(lines[start - 1 : end])

    def test_trailing_blank_lines_are_trimmed(self):
        lines = ["code", "", "   ", ""]
        assert trim_trailing_blank_lines(lines, 1, 4) == 1

    def test_rejects_non_positive_budget(self):
        with pytest.raises(ValueError):
            cap_span(["a"], 1, 1, 0)


class TestFixedChunkerAlignment:
    def test_trailing_blank_line_does_not_overstate_range(self):
        content = "alpha\nbeta\n\n"
        chunks = RecursiveChunker().chunk(content, "n.md")
        assert len(chunks) == 1
        assert chunks[0].end_line == 2
        assert_aligned(chunks[0], content)

    def test_windows_stay_aligned_across_a_large_file(self):
        content = "\n".join(f"row {index}" for index in range(1, 201))
        # 使用小块以测试分块行为（200 行每行约 10 字符 = 2000 字符，分成多块）
        chunks = RecursiveChunker(chunk_size=800, chunk_overlap=100).chunk(content, "big.md")
        assert len(chunks) > 1
        for chunk in chunks:
            assert_aligned(chunk, content)

    def test_long_lines_are_capped(self):
        content = "\n".join("y" * 3_000 for _ in range(8))
        chunks = RecursiveChunker(chunk_size=4000, chunk_overlap=200).chunk(
            content, "wide.md"
        )
        assert chunks
        assert all(len(chunk.content) <= 4_000 for chunk in chunks)


class TestCastChunkerAlignment:
    @pytest.mark.parametrize(
        "path,content",
        [
            (
                "src/app.ts",
                "export const handler = async (request: Request) => {\n"
                + "\n".join(f"  const step{index} = {index};" for index in range(60))
                + "\n  return step0;\n};\n",
            ),
            (
                "src/calc.py",
                "class Calculator:\n"
                + "\n".join(
                    f"    def method_{index}(self):\n        return {index}"
                    for index in range(40)
                )
                + "\n",
            ),
        ],
    )
    def test_ast_chunks_match_their_declared_lines(self, path, content):
        # The character budget also bounds how large a declaration may be and
        # still be kept whole, so it has to stay near the fixtures: at the
        # default 6000 each of these would come back as one chunk and the
        # alignment check would never see a boundary.
        chunker = CastChunker(
            max_chunk_size=300,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
            max_chunk_chars=1_000,
        )
        chunks = chunker.chunk(content, path)
        assert len(chunks) > 1
        for chunk in chunks:
            assert_aligned(chunk, content)

    def test_ast_chunks_respect_the_character_budget(self):
        content = "function build() {\n" + "\n".join(
            f"  const item{index} = {'z' * 120};" for index in range(200)
        ) + "\n}\n"
        chunker = CastChunker(
            max_chunk_size=100_000,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
            max_chunk_chars=5_000,
        )
        chunks = chunker.chunk(content, "src/build.ts")
        assert chunks
        assert all(len(chunk.content) <= 5_000 for chunk in chunks)
        for chunk in chunks:
            assert_aligned(chunk, content)

    def test_rejects_non_positive_char_budget(self):
        with pytest.raises(ValueError):
            CastChunker(
                max_chunk_size=1_500,
                chunk_overlap=0,
                fallback=RecursiveChunker(),
                max_chunk_chars=0,
            )

    def test_rejects_min_above_max(self):
        with pytest.raises(ValueError):
            CastChunker(
                max_chunk_size=1_500,
                chunk_overlap=0,
                fallback=RecursiveChunker(),
                max_chunk_chars=1_000,
                min_chunk_chars=1_000,
            )

    def test_generated_single_line_payload_yields_nothing(self):
        """单行生成产物解析得动，但每行都超预算，不能退回按字符切。"""
        content = (
            "const RAW = [\n"
            + "  '" + "a" * 8_000 + "',\n"
            + "  '" + "b" * 8_000 + "',\n"
            + "].join('');\n"
        )
        chunker = CastChunker(
            max_chunk_size=1_500,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
            max_chunk_chars=6_000,
        )
        chunks = chunker.chunk(content, "src/bundled.generated.ts")
        assert all(len(chunk.content) <= 6_000 for chunk in chunks)
        assert all(chunk.chunk_type != "recursive" for chunk in chunks)
        for chunk in chunks:
            assert_aligned(chunk, content)
