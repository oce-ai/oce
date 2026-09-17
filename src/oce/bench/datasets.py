"""数据集注册表：alias -> 查询集 jsonl + 被测仓库元数据 + 本地仓库定位。

取代 bench_orchestrate.py 里硬编码的绝对路径（``FLASK_REPO = C:\\Users\\...\\flask``、
``QUERIES = BENCH_REPO / "benchmarks" / ...``）。本模块**不含任何绝对路径**：

- 数据集（``*.jsonl`` + ``*.metadata.json``）源在仓库根 ``bench/datasets/``，构建 wheel 时
  由 hatchling force-include 进包（``oce/bench/datasets/``）。运行期由
  ``default_datasets_dir()`` 按 ① 包内（安装版）② 仓库根（checkout）顺序解析。
- 被测仓库是**外部大仓**（flask/cc-switch 的真实 clone），不进包；本地路径按
  显式 ``--repo-root`` > 环境变量 > cwd 同级目录约定 逐级解析，解析不到就明确报错让用户传。

metadata.json 的 schema（见 oce-benchmark）：
``{schema_version, benchmark, questions, repository: {name, remote, commit, describe}}``
—— commit/describe 是数据集出题时钉住的被测仓库版本，用于 RunRecord 溯源与"lock 状态"判断。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

# 本地被测仓库根的环境变量前缀：OCE_BENCH_REPO_FLASK=/path/to/flask
_REPO_ENV_PREFIX = "OCE_BENCH_REPO_"


class DatasetError(Exception):
    """数据集解析 / 仓库定位失败。消息面向用户，给出可用的补救动作。"""


@dataclass(frozen=True)
class RepositorySpec:
    """数据集出题时钉住的被测仓库版本（来自 metadata.json 的 repository 段）。"""

    name: str
    remote: str = ""
    commit: str | None = None
    describe: str = ""


@dataclass(frozen=True)
class Dataset:
    """一个可用的评测数据集：查询集 + 仓库元数据 + 题数。"""

    alias: str  # 无后缀文件名，如 "flask-retrieval-benchmark"
    name: str  # 仓库名，如 "flask"（--repo 用的短别名）
    queries_path: Path
    metadata_path: Path | None
    questions: int
    repository: RepositorySpec


def default_datasets_dir() -> Path:
    """定位数据集目录，两种安装形态都支持，无绝对路径。

    1. 包内（wheel / uv tool install）：构建时 hatchling 把仓库根 bench/datasets/
       force-include 成 oce/bench/datasets/；importlib.resources 负责按安装布局定位。
    2. checkout（仓库内 uv run 开发）：editable 安装不带 force-include 拷贝，从
       __file__ 上溯 3 级到仓库根的 bench/datasets/。

    返回第一个真实含 *.jsonl 的目录；都不存在时返回包内路径（让调用方报错指向正位）。
    """
    pkg_dir = Path(__file__).resolve().parent / "datasets"
    try:
        from importlib.resources import files

        pkg_dir = Path(str(files("oce.bench"))) / "datasets"
    except (ImportError, TypeError, AttributeError):
        pass  # 异常安装布局时 files() 可能给不出路径，回落 __file__
    checkout_dir = Path(__file__).resolve().parents[3] / "bench" / "datasets"
    for candidate in (pkg_dir, checkout_dir):
        if candidate.is_dir() and any(candidate.glob("*.jsonl")):
            return candidate
    return pkg_dir


def discover_datasets(datasets_dir: Path | None = None) -> list[Dataset]:
    """扫描目录下所有 ``*.jsonl``，配对 ``*.metadata.json``，按 alias 排序返回。

    没有 metadata 的 jsonl 也收（questions 现场数行数，repository 用文件名兜底）——宽容，
    避免一个缺 metadata 的数据集让整个 list 失败。
    """
    directory = Path(datasets_dir) if datasets_dir is not None else default_datasets_dir()
    if not directory.is_dir():
        return []
    datasets: list[Dataset] = []
    for queries_path in sorted(directory.glob("*.jsonl")):
        alias = queries_path.name[: -len(".jsonl")]
        metadata_path = directory / f"{alias}.metadata.json"
        metadata = _read_metadata(metadata_path)
        if metadata is not None:
            repo = RepositorySpec(
                name=metadata.get("repository", {}).get("name", alias),
                remote=metadata.get("repository", {}).get("remote", ""),
                commit=metadata.get("repository", {}).get("commit"),
                describe=metadata.get("repository", {}).get("describe", ""),
            )
            questions = int(metadata.get("questions", 0)) or _count_lines(queries_path)
        else:
            repo = RepositorySpec(name=alias)
            questions = _count_lines(queries_path)
        datasets.append(
            Dataset(
                alias=alias,
                name=repo.name,
                queries_path=queries_path,
                metadata_path=metadata_path if metadata is not None else None,
                questions=questions,
                repository=repo,
            )
        )
    return datasets


def _read_metadata(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _count_lines(path: Path) -> int:
    """数 jsonl 非空行数（metadata 缺失时的题数兜底）。"""
    try:
        with path.open(encoding="utf-8-sig") as handle:
            return sum(1 for line in handle if line.strip())
    except OSError:
        return 0


def find_dataset(
    ref: str, datasets_dir: Path | None = None
) -> Dataset:
    """按 ``ref`` 解析一个数据集：先精确 alias，再仓库短名，再唯一前缀匹配。

    - ``flask-retrieval-benchmark`` -> 精确 alias 命中。
    - ``flask`` -> 仓库短名命中（若唯一）。
    - ``flask-retr`` -> 唯一前缀命中（方便少打几个字）。
    多个候选或零候选都报错，列出可用项。
    """
    available = discover_datasets(datasets_dir)
    if not available:
        raise DatasetError(
            "no datasets found; expected *.jsonl in "
            f"{datasets_dir or default_datasets_dir()}"
        )
    # 1) 精确 alias
    for ds in available:
        if ds.alias == ref:
            return ds
    # 2) 仓库短名
    name_hits = [ds for ds in available if ds.name == ref]
    if len(name_hits) == 1:
        return name_hits[0]
    if len(name_hits) > 1:
        raise DatasetError(
            f"'{ref}' matches multiple datasets by name: "
            f"{[d.alias for d in name_hits]}; use the full alias"
        )
    # 3) 唯一前缀
    prefix_hits = [ds for ds in available if ds.alias.startswith(ref)]
    if len(prefix_hits) == 1:
        return prefix_hits[0]
    if len(prefix_hits) > 1:
        raise DatasetError(
            f"'{ref}' is ambiguous (prefix match): {[d.alias for d in prefix_hits]}"
        )
    raise DatasetError(
        f"dataset '{ref}' not found; available: {[d.alias for d in available]}"
    )


def resolve_repo_root(
    dataset: Dataset, *, explicit: str | None = None
) -> Path:
    """定位被测仓库的本地 clone（外部大仓，不进包）。

    优先级：① 显式 ``--repo-root`` ② 环境变量 ``OCE_BENCH_REPO_<NAME 大写>``（连字符转下划线）
    ③ cwd 及其父目录下名为 ``<repo name>`` 的同级目录。都落空则报错，指导用户传 --repo-root
    或设环境变量——绝不回退到任何硬编码绝对路径。
    """
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise DatasetError(f"--repo-root is not a directory: {path}")
        return path

    env_name = _REPO_ENV_PREFIX + dataset.name.upper().replace("-", "_")
    env_value = os.environ.get(env_name)
    if env_value:
        path = Path(env_value).expanduser().resolve()
        if not path.is_dir():
            raise DatasetError(f"{env_name}={env_value} is not a directory")
        return path

    # cwd 同级约定：./<name> 或 ../<name>（在 oce 仓库里跑时，flask 常是其同级目录）
    cwd = Path.cwd()
    for candidate in (cwd / dataset.name, cwd.parent / dataset.name):
        if candidate.is_dir():
            return candidate.resolve()

    raise DatasetError(
        f"cannot locate local clone of '{dataset.name}'; pass --repo-root <path> "
        f"or set {env_name}=<path> (remote: {dataset.repository.remote or 'unknown'})"
    )
