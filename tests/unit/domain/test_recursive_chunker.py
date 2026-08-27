"""RecursiveChunker 测试"""
import pytest

from oce.domain.chunk import RecursiveChunker, is_meaningful
from oce.domain.chunk.types import Chunk


class TestRecursiveChunker:
    """测试 RecursiveChunker 的基本功能"""

    def test_empty_content(self):
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
        assert chunker.chunk("", "test.py") == []

    def test_meaningless_content(self):
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
        content = "   \n\n  \t  \n   "
        assert chunker.chunk(content, "test.py") == []

    def test_small_file_single_chunk(self):
        """小文件应该产生单个 chunk"""
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
        content = "def hello():\n    print('world')\n"
        chunks = chunker.chunk(content, "test.py")

        assert len(chunks) == 1
        # RecursiveChunker 可能会去掉尾部空行
        assert chunks[0].content.strip() == content.strip()
        assert chunks[0].path == "test.py"
        assert chunks[0].chunk_type == "recursive"
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 2

    def test_chunk_splitting_by_paragraphs(self):
        """测试按段落分隔（内容足够大时）"""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        # 需要足够长的内容才会触发分割
        content = "Line 1 with more content\nLine 2 with more content\n\nLine 3 with more content\nLine 4 with more content\n\nLine 5 with more content\nLine 6 with more content"
        chunks = chunker.chunk(content, "test.txt")

        # 应该在 \n\n 处优先切分（如果内容够大）
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.path == "test.txt"
            assert chunk.chunk_type == "recursive"

    def test_single_line_is_not_split(self):
        """单行内容不可拆分：一行拆成两块，就没法各自声明正确的行号。

        切分只在行边界发生，所以超过 chunk_size 的单行整行保留，而不是像旧实现
        那样按字符切成几片、每片都声明同一行。
        """
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        content = "A" * 200
        chunks = chunker.chunk(content, "test.txt")

        assert len(chunks) == 1
        assert chunks[0].content == content
        assert (chunks[0].start_line, chunks[0].end_line) == (1, 1)

    def test_chunks_do_not_overlap(self):
        """带行号的 chunk 之间不能重叠：同样的行索引两遍，占两个检索位。"""
        chunker = RecursiveChunker(chunk_size=60, chunk_overlap=20)
        content = "\n".join(f"line {index} carries some content" for index in range(40))
        chunks = chunker.chunk(content, "notes.txt")

        assert len(chunks) > 1
        for previous, current in zip(chunks, chunks[1:]):
            assert current.start_line > previous.end_line

    def test_python_language_aware(self):
        """测试 Python 语言特定的分隔符"""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        content = """def func1():
    pass

def func2():
    pass

def func3():
    pass"""
        chunks = chunker.chunk(content, "test.py")
        
        # Python 分隔符应该优先在函数之间切分
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "def " in chunk.content

    def test_chunk_hash_uniqueness(self):
        """测试不同内容产生不同的 hash"""
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
        chunks1 = chunker.chunk("content A", "test.txt")
        chunks2 = chunker.chunk("content B", "test.txt")
        
        assert chunks1[0].content_hash != chunks2[0].content_hash

    def test_line_number_accuracy(self):
        """测试行号计算准确性"""
        chunker = RecursiveChunker(chunk_size=1000, chunk_overlap=100)
        content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        chunks = chunker.chunk(content, "test.txt")
        
        assert len(chunks) == 1
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 5

    def test_unsupported_language_fallback(self):
        """测试不支持的语言使用通用分隔符"""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        content = "Some random content\n\nMore content"
        chunks = chunker.chunk(content, "unknown.xyz")
        
        assert len(chunks) >= 1
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_large_file_multiple_chunks(self):
        """测试大文件被正确分割"""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        content = "\n".join([f"Line {i}" for i in range(100)])
        chunks = chunker.chunk(content, "large.txt")

        assert len(chunks) > 1
        # 验证所有 chunk 大小在限制内（允许少量超出因为要保持完整性）
        for chunk in chunks:
            assert len(chunk.content) <= 200  # 允许一些弹性

    @pytest.mark.parametrize(
        "path,content",
        [
            # 缩进内容：splitter 会 strip 掉分隔符处的空白，块首缩进曾因此丢失，
            # 令文本不再等于它声明的那些行。
            (
                "styles.css",
                "@media (max-width: 600px) {\n"
                + "\n".join(
                    f"  .item-{index} {{\n    padding: {index}px;\n  }}"
                    for index in range(30)
                )
                + "\n}\n",
            ),
            (
                "data.json",
                "{\n"
                + ",\n".join(f'  "key{index}": "value{index}"' for index in range(60))
                + "\n}\n",
            ),
            ("notes.txt", "\n\n".join(f"段落 {index} 的正文内容。" for index in range(30))),
        ],
    )
    def test_chunks_match_their_declared_lines(self, path, content):
        """chunk 文本必须等于它声明的源码行 —— formatter 按 start_line + offset 渲染。"""
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=20)
        lines = content.splitlines()
        chunks = chunker.chunk(content, path)

        assert chunks
        for chunk in chunks:
            expected = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
            assert chunk.content == expected, (
                f"{path}#{chunk.start_line}-{chunk.end_line} 文本与声明的行范围不符"
            )

    def test_overlong_single_line_is_dropped(self):
        """超过硬上限的单行无法在保持行号正确的前提下切分，只能丢弃。"""
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=0)
        content = "head line\n" + "z" * 7_000 + "\ntail line\n"
        chunks = chunker.chunk(content, "bundle.min.js")

        assert all(len(chunk.content) <= 6_000 for chunk in chunks)
        assert all("z" * 100 not in chunk.content for chunk in chunks)
        lines = content.splitlines()
        for chunk in chunks:
            assert chunk.content == "\n".join(
                lines[chunk.start_line - 1 : chunk.end_line]
            )


class TestIsMeaningful:
    """测试 is_meaningful 辅助函数"""

    def test_meaningful_content(self):
        assert is_meaningful("hello")
        assert is_meaningful("123")
        assert is_meaningful("  abc  ")
        assert is_meaningful("符号123")

    def test_meaningless_content(self):
        assert not is_meaningful("")
        assert not is_meaningful("   ")
        assert not is_meaningful("\n\n\n")
        assert not is_meaningful(".,;!?")
        assert not is_meaningful("   \t\n   ")
