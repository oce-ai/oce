"""本地仓库遍历 + 上传批次切分。

两处刻意复用 oce 主体、而非从 harness 搬运复制品：

1. **内容寻址**用 ``application.service.compute_blob_name``（``sha256(path + content)``）。
   harness 里有一份一模一样的复制品；两份算法一旦漂移，``--reuse-index`` 的
   ``/find-missing`` 会把已存在的 blob 误报为缺失（或反之），静默毁掉索引复用。
2. **准入规则**用 ``domain.services.source_filter.is_ignored_source_path``。harness 自带
   一份 ``IGNORED_DIRECTORIES`` / ``IGNORED_SUFFIXES``，比服务端清单旧（缺 ``.venv`` /
   ``__pycache__`` / ``.jsonl`` / ``.csv`` 等）。若本地遍历用 A 清单、服务端索引用 B 清单，
   上传集与索引集就会不一致。复用同一函数从根上杜绝这类漂移。

harness 独有、主体没有的两条本地规则在此保留：跳过 ``.env*``（评测客户端绝不上传密钥）、
跳过超过 ``MAX_FILE_BYTES`` 的巨文件与二进制（NUL 字节）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from oce.application.service import compute_blob_name
from oce.domain.services.source_filter import (
    is_binary_source,
    is_ignored_source_path,
)

# 单文件读取上限（超过则跳过，避免巨文件撑爆上传/嵌入）
MAX_FILE_BYTES = 1_000_000
# 上传批次切分阈值：文件数或字节数任一超限即切批
MAX_BATCH_FILES = 32
MAX_BATCH_BYTES = 1_500_000

# 评测客户端绝不上传环境文件（可能含密钥）；这是本地规则，与主体准入正交。
_SECRET_FILENAMES = frozenset({".env", ".env.local", ".env.production"})

# is_ignored_source_path 把路径**最后一段当文件名**、其余段当目录（parts[:-1]）。要判一个
# 目录是否该剪枝，必须塞一个 dummy leaf 让目录名落进 parts[:-1]，否则目录名会被当文件名、
# 漏过 IGNORED_DIRECTORY_NAMES 判定（实测 "node_modules/" -> False，"node_modules/x" -> True）。
# leaf 选 "__probe__"：不含点、不以任何 IGNORED_FILE_SUFFIXES 结尾，绝不误命中文件后缀分支。
_DIR_PROBE_LEAF = "__probe__"


def _is_ignored_dir(rel_dir: str) -> bool:
    """判定一个仓库相对目录是否该被剪枝（复用 source_filter 的目录规则）。"""
    return is_ignored_source_path(rel_dir.rstrip("/") + "/" + _DIR_PROBE_LEAF)


@dataclass(frozen=True)
class SourceBlob:
    """一个待上传的源文件：仓库相对路径（posix 风格）+ utf-8 文本内容。"""

    path: str
    content: str

    @property
    def blob_name(self) -> str:
        """内容寻址名，与服务端一致（复用同一 compute_blob_name）。"""
        return compute_blob_name(self.path, self.content)


def iter_source_blobs(repo_root: Path) -> Iterator[SourceBlob]:
    """逐个产出仓库里可读的文本源文件（惰性，不一次性载入整仓）。

    跳过：依赖/生成/非源码目录与后缀（复用 source_filter）、``.env*``、超 ``MAX_FILE_BYTES``
    的文件、二进制（NUL 字节）、非 utf-8。相对路径统一用 posix 分隔符，使评分与数据集里的
    ``expected_files`` 口径一致。
    """
    for directory, dirnames, filenames in os.walk(repo_root):
        # 就地裁剪 dirnames，让 os.walk 不进入被忽略目录（剪枝而非事后过滤，省时）。
        # 目录判定走 _is_ignored_dir（probe leaf），文件判定走 is_ignored_source_path。
        base = Path(directory).relative_to(repo_root)
        dirnames[:] = [
            name
            for name in dirnames
            if not _is_ignored_dir((base / name).as_posix())
        ]
        for filename in filenames:
            if filename in _SECRET_FILENAMES:
                continue
            path = Path(directory) / filename
            relative = path.relative_to(repo_root).as_posix()
            if is_ignored_source_path(relative):
                continue
            try:
                size = path.stat().st_size
                if size > MAX_FILE_BYTES:
                    continue
                raw = path.read_bytes()
            except (OSError, ValueError):
                continue
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if is_binary_source(content):
                continue
            yield SourceBlob(relative, content)


def make_batches(blobs: Iterator[SourceBlob]) -> Iterator[list[SourceBlob]]:
    """把惰性 blob 流切成上传批次：文件数或累计字节任一超限即切。

    单个文件超 ``MAX_BATCH_BYTES`` 时它独占一批（不会无限切分），交由服务端处理。
    """
    batch: list[SourceBlob] = []
    batch_bytes = 0
    for blob in blobs:
        blob_bytes = len(blob.content.encode("utf-8"))
        if batch and (
            len(batch) >= MAX_BATCH_FILES
            or batch_bytes + blob_bytes > MAX_BATCH_BYTES
        ):
            yield batch
            batch = []
            batch_bytes = 0
        batch.append(blob)
        batch_bytes += blob_bytes
    if batch:
        yield batch
