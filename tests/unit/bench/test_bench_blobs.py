"""本地仓库遍历 + 批次切分测试（tmp_path，零外部依赖）。

重点验证 blobs.py 复用主体准入规则（source_filter）与内容寻址（compute_blob_name）的正确性
—— 这是 --reuse-index 不误报的前提。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from oce.application.service import compute_blob_name
from oce.bench.blobs import (
    MAX_BATCH_BYTES,
    MAX_BATCH_FILES,
    MAX_FILE_BYTES,
    SourceBlob,
    iter_source_blobs,
    make_batches,
)


def _write(root: Path, rel: str, content: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_bytes(content.encode("utf-8"))


class TestIterSourceBlobs:
    def test_collects_text_sources_with_posix_paths(self, tmp_path):
        _write(tmp_path, "src/app.py", "print(1)\n")
        _write(tmp_path, "pyproject.toml", "[project]\n")
        paths = sorted(b.path for b in iter_source_blobs(tmp_path))
        assert paths == ["pyproject.toml", "src/app.py"]

    def test_prunes_ignored_directories(self, tmp_path):
        """node_modules / .git / .venv 等被剪枝（复用 source_filter 目录规则）。"""
        _write(tmp_path, "src/app.py", "x")
        _write(tmp_path, "node_modules/lib.js", "x")
        _write(tmp_path, ".git/config", "x")
        _write(tmp_path, ".venv/site.py", "x")
        _write(tmp_path, "__pycache__/m.pyc", "x")
        paths = sorted(b.path for b in iter_source_blobs(tmp_path))
        assert paths == ["src/app.py"]

    def test_skips_ignored_suffixes(self, tmp_path):
        _write(tmp_path, "src/app.py", "x")
        _write(tmp_path, "data.jsonl", "{}")
        _write(tmp_path, "rows.csv", "a,b")
        _write(tmp_path, "styles.min.css", "x")
        paths = sorted(b.path for b in iter_source_blobs(tmp_path))
        assert paths == ["src/app.py"]

    def test_skips_env_files(self, tmp_path):
        """.env* 绝不上传（可能含密钥）——harness 独有的本地安全规则。"""
        _write(tmp_path, "src/app.py", "x")
        _write(tmp_path, ".env", "SECRET=leak")
        _write(tmp_path, ".env.local", "SECRET=leak")
        _write(tmp_path, ".env.production", "SECRET=leak")
        paths = sorted(b.path for b in iter_source_blobs(tmp_path))
        assert paths == ["src/app.py"]

    def test_skips_oversized_files(self, tmp_path):
        _write(tmp_path, "small.py", "x")
        _write(tmp_path, "big.txt", "z" * (MAX_FILE_BYTES + 1))
        paths = sorted(b.path for b in iter_source_blobs(tmp_path))
        assert paths == ["small.py"]

    def test_skips_binary_and_non_utf8(self, tmp_path):
        _write(tmp_path, "src/app.py", "x")
        _write(tmp_path, "bin.dat", b"\x00\x01\x02")
        _write(tmp_path, "latin.txt", b"\xff\xfe\xfd")  # 非法 utf-8
        paths = sorted(b.path for b in iter_source_blobs(tmp_path))
        assert paths == ["src/app.py"]

    def test_keeps_empty_file(self, tmp_path):
        """空 __init__.py 应保留（是真实源文件，非二进制）。"""
        _write(tmp_path, "pkg/__init__.py", "")
        paths = [b.path for b in iter_source_blobs(tmp_path)]
        assert paths == ["pkg/__init__.py"]

    def test_blob_name_matches_main_compute(self, tmp_path):
        """blob_name 复用主体 compute_blob_name：内容寻址一致，--reuse-index 才不误报。"""
        _write(tmp_path, "src/app.py", "print(1)\n")
        blob = next(b for b in iter_source_blobs(tmp_path) if b.path == "src/app.py")
        assert blob.blob_name == compute_blob_name("src/app.py", "print(1)\n")

    def test_is_lazy_generator(self, tmp_path):
        """iter_source_blobs 返回生成器（惰性，不一次性载入整仓）。"""
        import types

        _write(tmp_path, "a.py", "x")
        assert isinstance(iter_source_blobs(tmp_path), types.GeneratorType)


class TestMakeBatches:
    def test_splits_on_file_count(self):
        blobs = (SourceBlob(f"f{i}.py", "y" * 10) for i in range(100))
        sizes = [len(b) for b in make_batches(blobs)]
        assert sum(sizes) == 100
        assert all(s <= MAX_BATCH_FILES for s in sizes)

    def test_splits_on_byte_budget(self):
        # 每个 600KB，预算 1.5MB -> 每批最多 2 个
        blobs = (SourceBlob(f"g{i}.txt", "z" * 600_000) for i in range(5))
        sizes = [len(b) for b in make_batches(blobs)]
        assert sum(sizes) == 5
        assert all(s <= 2 for s in sizes)

    def test_single_oversized_file_gets_own_batch(self):
        blob = SourceBlob("huge.txt", "q" * (MAX_BATCH_BYTES + 100))
        batches = list(make_batches(iter([blob])))
        assert len(batches) == 1 and len(batches[0]) == 1

    def test_empty_stream_yields_nothing(self):
        assert list(make_batches(iter([]))) == []

    def test_preserves_order(self):
        blobs = (SourceBlob(f"f{i}.py", "y") for i in range(5))
        flat = [b.path for batch in make_batches(blobs) for b in batch]
        assert flat == [f"f{i}.py" for i in range(5)]
