"""Chunker 领域服务测试"""

from __future__ import annotations

import pytest

from oce.domain.chunk import RecursiveChunker
from oce.infrastructure.astchunk.cast_chunker import CastChunker


class TestCASTChunker:
    """cAST 语义切块"""

    def test_document_formats_should_not_be_in_cast_chunker_languages(self):
        """文档格式（markdown/jsp/vue/svelte）不应该在 CastChunker 的 languages 中"""
        cast_chunker = CastChunker(
            max_chunk_size=500,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
        )

        document_formats = {"markdown", "jsp", "vue", "svelte"}
        assert document_formats.isdisjoint(cast_chunker.languages)

    def test_programming_languages_are_in_cast_chunker(self):
        """编程语言（python/java/ts等）应该在 CastChunker 的 languages 中"""
        cast_chunker = CastChunker(
            max_chunk_size=500,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
        )

        # 有 tree-sitter parser 的编程语言
        programming_languages = {"python", "java", "javascript", "typescript", "go", "rust", "swift", "kotlin"}
        assert programming_languages.issubset(cast_chunker.languages)

    def test_unsupported_language_falls_back_to_recursive(self):
        content = "\n".join(f"line{i}" for i in range(100))
        fallback = RecursiveChunker(chunk_size=6000, chunk_overlap=200)
        chunker = CastChunker(max_chunk_size=1500, chunk_overlap=0, fallback=fallback)
        chunks = chunker.chunk(content, "notes.unknown_ext")
        # 回退 recursive：应该被切分
        assert len(chunks) >= 1

    def test_python_file_chunks_with_ast(self):
        content = "\n".join(
            f"def func_{i}():\n    return {i}" for i in range(50)
        )
        fallback = RecursiveChunker(chunk_size=6000, chunk_overlap=200)
        chunker = CastChunker(max_chunk_size=500, chunk_overlap=0, fallback=fallback)
        chunks = chunker.chunk(content, "src/demo.py")
        assert len(chunks) >= 1
        # 行号 1-based
        assert all(c.start_line >= 1 for c in chunks)
        assert all(c.end_line >= c.start_line for c in chunks)

    def test_java_file_chunks_with_ast(self):
        content = """public class UserService {
    public String findName(long id) {
        return \"user-\" + id;
    }
}
"""
        chunks = CastChunker(
            max_chunk_size=500,
            chunk_overlap=0,
            fallback=RecursiveChunker(),
        ).chunk(content, "src/main/java/com/example/UserService.java")

        assert chunks
        assert all(chunk.chunk_type != "recursive" for chunk in chunks)
        assert "class UserService" in chunks[0].content

    def test_ast_end_line_is_inclusive_for_trailing_newline(self):
        content = "def hello():\n    return 'world'\n"
        fallback = RecursiveChunker(chunk_size=6000, chunk_overlap=200)
        chunks = CastChunker(
            max_chunk_size=1500,
            chunk_overlap=0,
            fallback=fallback,
        ).chunk(content, "src/hello.py")

        assert chunks
        assert chunks[0].chunk_type != "recursive"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2

    def test_empty_content(self):
        fallback = RecursiveChunker(chunk_size=6000, chunk_overlap=200)
        assert CastChunker(max_chunk_size=1500, chunk_overlap=0, fallback=fallback).chunk("", "src/empty.py") == []

    def test_chunk_path_preserved(self):
        content = "def hello():\n    return 'world'\n"
        fallback = RecursiveChunker(chunk_size=6000, chunk_overlap=200)
        chunks = CastChunker(max_chunk_size=1500, chunk_overlap=0, fallback=fallback).chunk(content, "src/hello.py")
        assert chunks
        assert all(c.path == "src/hello.py" for c in chunks)
