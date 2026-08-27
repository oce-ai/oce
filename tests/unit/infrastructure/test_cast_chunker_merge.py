"""CastChunker 块边界质量的契约测试。

两组不变量：
- 最小块合并：astchunk 会把同一个构造拆成起点相同的两个 window（第一个只含
  分割列之前的内容），合并要同时处理「被包含」和「过小」，且不能丢行。
- 声明保全：略微超出窗口的声明必须整块保留，否则递归会交回它的语句列表，
  贪心装箱再从中间切断，得到「有名字没实现」和「以 return { 开头」的两半。
"""

from __future__ import annotations

import pytest

from oce.domain.chunk.recursive_chunker import RecursiveChunker
from oce.infrastructure.astchunk.cast_chunker import CastChunker


def make_chunker(min_chunk_chars: int = 300) -> CastChunker:
    return CastChunker(
        max_chunk_size=1_500,
        chunk_overlap=0,
        fallback=RecursiveChunker(),
        min_chunk_chars=min_chunk_chars,
    )


def lines_of(count: int, width: int = 40) -> list[str]:
    return [f"line{index:03d}".ljust(width, "x") for index in range(1, count + 1)]


class TestMergeSmall:
    def test_range_sharing_a_start_line_is_absorbed(self):
        """astchunk 对同一构造给出的前缀 window 不应单独成块。"""
        lines = lines_of(40)
        ranges = [(1, 1, "ast"), (1, 20, "ast"), (21, 40, "ast")]
        merged = make_chunker()._merge_small(ranges, lines)
        assert (1, 1, "ast") not in merged
        assert merged == [(1, 20, "ast"), (21, 40, "ast")]

    def test_fully_contained_range_is_dropped(self):
        lines = lines_of(40)
        ranges = [(1, 30, "ast"), (5, 12, "ast"), (31, 40, "ast")]
        merged = make_chunker()._merge_small(ranges, lines)
        assert merged == [(1, 30, "ast"), (31, 40, "ast")]

    def test_trailing_fragment_attaches_to_previous_range(self):
        """孤立的收尾括号并入前一块，而不是自成一块。"""
        lines = lines_of(41)
        ranges = [(1, 40, "ast"), (41, 41, "ast")]
        merged = make_chunker()._merge_small(ranges, lines)
        assert merged == [(1, 41, "ast")]

    def test_leading_fragment_absorbs_the_next_range(self):
        """首块过小时向后吞并，保证第一块也带够上下文。"""
        lines = lines_of(40)
        ranges = [(1, 2, "ast"), (3, 30, "ast"), (31, 40, "ast")]
        merged = make_chunker()._merge_small(ranges, lines)
        assert merged[0] == (1, 30, "ast")

    def test_merging_preserves_line_coverage(self):
        lines = lines_of(60)
        ranges = [
            (1, 1, "ast"),
            (2, 3, "ast"),
            (4, 25, "ast"),
            (26, 26, "ast"),
            (27, 60, "ast"),
        ]
        merged = make_chunker()._merge_small(ranges, lines)
        covered: set[int] = set()
        for start, end, _ in merged:
            covered.update(range(start, end + 1))
        assert covered == set(range(1, 61))

    def test_zero_floor_disables_merging(self):
        lines = lines_of(40)
        ranges = [(1, 1, "ast"), (1, 20, "ast")]
        assert make_chunker(min_chunk_chars=0)._merge_small(ranges, lines) == ranges

    def test_chunk_type_of_the_kept_range_wins(self):
        lines = lines_of(40)
        ranges = [(1, 1, "class_declaration"), (1, 20, "function_declaration")]
        merged = make_chunker()._merge_small(ranges, lines)
        assert merged == [(1, 20, "function_declaration")]


class TestMergeThroughPublicApi:
    def test_signature_only_chunk_does_not_survive(self):
        """声明头和函数体被拆成两个 window 时，不应留下只有签名的块。"""
        content = (
            "export type UiState = {\n"
            + "\n".join(f"  field{index}: string;" for index in range(80))
            + "\n};\n"
            + "export function build(): UiState {\n"
            + "\n".join(f"  const step{index} = {index};" for index in range(80))
            + "\n  return null as unknown as UiState;\n}\n"
        )
        chunks = make_chunker().chunk(content, "src/ui.ts")
        assert chunks
        tiny = [chunk for chunk in chunks if len(chunk.content) < 300]
        assert tiny == [], [chunk.content for chunk in tiny]

    def test_merged_chunks_still_match_their_line_range(self):
        content = (
            "class Service {\n"
            + "\n".join(
                f"  method{index}() {{\n    return {index};\n  }}"
                for index in range(60)
            )
            + "\n}\n"
        )
        lines = content.splitlines()
        for chunk in make_chunker().chunk(content, "src/service.ts"):
            expected = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
            assert chunk.content == expected

    def test_merged_chunks_stay_within_the_char_budget(self):
        content = "function run() {\n" + "\n".join(
            f"  const value{index} = {'q' * 90};" for index in range(200)
        ) + "\n}\n"
        chunker = CastChunker(
            max_chunk_size=100_000,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
            max_chunk_chars=4_000,
            min_chunk_chars=300,
        )
        chunks = chunker.chunk(content, "src/run.ts")
        assert chunks
        assert all(len(chunk.content) <= 4_000 for chunk in chunks)


class TestIntactDeclarations:
    """略微超窗的声明不应被拆成「签名」和「孤立主体」两半。"""

    @staticmethod
    def _oversized_case(name: str, statements: int = 40) -> str:
        body = "\n".join(
            f'    expect(result.field{index}).toBe("value{index}");'
            for index in range(statements)
        )
        return f'  it("{name}", async () => {{\n{body}\n  }});\n'

    def test_oversized_test_case_stays_whole(self):
        content = (
            'describe("suite", () => {\n'
            + self._oversized_case("handles the first branch")
            + self._oversized_case("handles the second branch")
            + "});\n"
        )
        chunker = CastChunker(
            max_chunk_size=300,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
        )
        chunks = chunker.chunk(content, "src/suite.test.ts")
        bodies = [chunk.content for chunk in chunks if "it(" in chunk.content]
        assert bodies, [chunk.content[:60] for chunk in chunks]
        for body in bodies:
            # 带 it( 的块必须自带主体和收尾，而不是只剩一行签名
            assert body.count("expect(") > 1, body[:120]

    def test_declaration_without_field_names_stays_whole(self):
        """Kotlin 语法不给任何子节点命名字段，只能按子节点类型识别 body。

        按节点类型名列白名单时这里会退化：Kotlin 的声明类型不在 TypeScript
        推导出的那张表里，超窗后被静默拆开。
        """
        members = "\n".join(
            f'    fun member{index}(): String = "value{index}"'
            for index in range(40)
        )
        content = f"class Repository {{\n{members}\n}}\n"
        chunker = CastChunker(
            max_chunk_size=300,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
        )
        chunks = chunker.chunk(content, "src/Repository.kt")
        assert len(chunks) == 1, [chunk.content[:60] for chunk in chunks]
        assert chunks[0].content.startswith("class Repository {")
        assert chunks[0].content.rstrip().endswith("}")

    def test_character_budget_bounds_what_is_kept_whole(self):
        """保全上限跟着字符预算走，否则保住的块又被 cap_span 切回两半。"""
        content = (
            'describe("suite", () => {\n'
            + self._oversized_case("single case")
            + "});\n"
        )
        roomy = CastChunker(
            max_chunk_size=300,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
        ).chunk(content, "src/suite.test.ts")
        tight = CastChunker(
            max_chunk_size=300,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
            max_chunk_chars=800,
        ).chunk(content, "src/suite.test.ts")
        assert len(tight) > len(roomy)

    def test_runaway_wrapper_is_still_split(self):
        """整个文件包在一个 describe 里时，上限必须让它继续拆。"""
        content = (
            'describe("giant", () => {\n'
            + "".join(self._oversized_case(f"case {index}") for index in range(12))
            + "});\n"
        )
        chunker = CastChunker(
            max_chunk_size=300,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
        )
        chunks = chunker.chunk(content, "src/giant.test.ts")
        assert len(chunks) > 1
        lines = content.splitlines()
        for chunk in chunks:
            assert chunk.content == "\n".join(
                lines[chunk.start_line - 1 : chunk.end_line]
            )
