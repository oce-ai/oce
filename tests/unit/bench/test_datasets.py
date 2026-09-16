"""datasets.py 注册表测试：发现 / metadata 配对 / ref 解析 / 仓库定位（无硬编码绝对路径）。

核心断言：① discover_datasets 扫 *.jsonl 配对 *.metadata.json，缺 metadata 也宽容收（数行）
② find_dataset 三级匹配：精确 alias > 仓库短名 > 唯一前缀；歧义/零命中报错
③ resolve_repo_root 优先级：显式 --repo-root > 环境变量 > cwd 同级；落空报错且**绝不回退到
   任何硬编码路径**（这是取代 bench_orchestrate.py 里 FLASK_REPO=C:\\... 的关键）。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oce.bench.datasets import (
    Dataset,
    DatasetError,
    RepositorySpec,
    _count_lines,
    discover_datasets,
    find_dataset,
    resolve_repo_root,
)


def _write_dataset(
    directory: Path,
    alias: str,
    *,
    repo_name: str,
    questions: int | None = None,
    commit: str = "abc123def456",
    describe: str = "v1.0",
    remote: str = "https://example.com/repo.git",
    with_metadata: bool = True,
) -> Path:
    """造一个数据集：queries.jsonl +（可选）metadata.json。返回 jsonl 路径。"""
    directory.mkdir(parents=True, exist_ok=True)
    n = questions if questions is not None else 2
    queries = directory / f"{alias}.jsonl"
    lines = [
        json.dumps({"id": f"Q{i}", "category": "c", "difficulty": 1,
                    "query": "q", "expected_files": ["a.py"]})
        for i in range(n)
    ]
    queries.write_text("\n".join(lines), encoding="utf-8")
    if with_metadata:
        meta = directory / f"{alias}.metadata.json"
        meta.write_text(
            json.dumps({
                "schema_version": 1,
                "benchmark": alias,
                "questions": n,
                "repository": {
                    "name": repo_name, "remote": remote,
                    "commit": commit, "describe": describe,
                },
            }),
            encoding="utf-8",
        )
    return queries


# ---------------------------------------------------------------------------
# discover_datasets
# ---------------------------------------------------------------------------


def test_discover_pairs_metadata(tmp_path: Path):
    _write_dataset(tmp_path, "flask-retrieval-benchmark", repo_name="flask", questions=100)
    found = discover_datasets(tmp_path)
    assert len(found) == 1
    ds = found[0]
    assert ds.alias == "flask-retrieval-benchmark"
    assert ds.name == "flask"
    assert ds.questions == 100
    assert ds.repository.commit == "abc123def456"
    assert ds.repository.describe == "v1.0"
    assert ds.metadata_path is not None


def test_discover_without_metadata_counts_lines(tmp_path: Path):
    """缺 metadata 也宽容收：questions 现场数行，repository.name 用 alias 兜底。"""
    _write_dataset(
        tmp_path, "lonely-set", repo_name="ignored", questions=3, with_metadata=False
    )
    found = discover_datasets(tmp_path)
    assert len(found) == 1
    ds = found[0]
    assert ds.metadata_path is None
    assert ds.questions == 3
    assert ds.name == "lonely-set"
    assert ds.repository.commit is None


def test_discover_sorted_and_multiple(tmp_path: Path):
    _write_dataset(tmp_path, "zeta-set", repo_name="zeta")
    _write_dataset(tmp_path, "alpha-set", repo_name="alpha")
    found = discover_datasets(tmp_path)
    assert [d.alias for d in found] == ["alpha-set", "zeta-set"]


def test_discover_missing_dir_returns_empty(tmp_path: Path):
    assert discover_datasets(tmp_path / "absent") == []


def test_count_lines_ignores_blank(tmp_path: Path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n\n   \n{"b":2}\n', encoding="utf-8")
    assert _count_lines(p) == 2


def test_discover_corrupt_metadata_falls_back(tmp_path: Path):
    """metadata 解析失败 -> 当成无 metadata（数行兜底），不让整个 list 崩。"""
    _write_dataset(tmp_path, "broken", repo_name="b", questions=2, with_metadata=False)
    (tmp_path / "broken.metadata.json").write_text("{ not json", encoding="utf-8")
    found = discover_datasets(tmp_path)
    assert len(found) == 1
    assert found[0].metadata_path is None
    assert found[0].questions == 2


# ---------------------------------------------------------------------------
# find_dataset：三级匹配
# ---------------------------------------------------------------------------


@pytest.fixture
def two_datasets(tmp_path: Path) -> Path:
    _write_dataset(tmp_path, "flask-retrieval-benchmark", repo_name="flask")
    _write_dataset(tmp_path, "cc-switch-retrieval-benchmark", repo_name="cc-switch")
    return tmp_path


def test_find_by_exact_alias(two_datasets: Path):
    ds = find_dataset("flask-retrieval-benchmark", two_datasets)
    assert ds.name == "flask"


def test_find_by_repo_short_name(two_datasets: Path):
    ds = find_dataset("flask", two_datasets)
    assert ds.alias == "flask-retrieval-benchmark"


def test_find_by_unique_prefix(two_datasets: Path):
    ds = find_dataset("cc-switch-retr", two_datasets)
    assert ds.name == "cc-switch"


def test_find_ambiguous_name_raises(tmp_path: Path):
    """两个数据集 repo_name 都是 'flask' -> 短名歧义报错。"""
    _write_dataset(tmp_path, "flask-a", repo_name="flask")
    _write_dataset(tmp_path, "flask-b", repo_name="flask")
    with pytest.raises(DatasetError, match="multiple datasets by name"):
        find_dataset("flask", tmp_path)


def test_find_single_char_unique_prefix(two_datasets: Path):
    # "f" 只前缀命中 flask-...（cc-switch 不以 f 开头）-> 唯一前缀命中
    ds = find_dataset("f", two_datasets)
    assert ds.name == "flask"


def test_find_ambiguous_prefix_raises(tmp_path: Path):
    # 两个 alias 都以 "shared-" 开头 -> 前缀 "shared" 歧义
    _write_dataset(tmp_path, "shared-alpha", repo_name="alpha")
    _write_dataset(tmp_path, "shared-beta", repo_name="beta")
    with pytest.raises(DatasetError, match="ambiguous"):
        find_dataset("shared", tmp_path)


def test_find_unknown_raises_lists_available(two_datasets: Path):
    with pytest.raises(DatasetError, match="not found") as ei:
        find_dataset("nope", two_datasets)
    assert "flask-retrieval-benchmark" in str(ei.value)


def test_find_empty_dir_raises(tmp_path: Path):
    with pytest.raises(DatasetError, match="no datasets found"):
        find_dataset("anything", tmp_path)


# ---------------------------------------------------------------------------
# resolve_repo_root：优先级 + 无硬编码兜底
# ---------------------------------------------------------------------------


def _ds(name: str, remote: str = "https://example.com/x.git") -> Dataset:
    return Dataset(
        alias=f"{name}-set", name=name, queries_path=Path(f"{name}.jsonl"),
        metadata_path=None, questions=1,
        repository=RepositorySpec(name=name, remote=remote),
    )


def test_resolve_explicit_repo_root(tmp_path: Path):
    target = tmp_path / "flask-clone"
    target.mkdir()
    resolved = resolve_repo_root(_ds("flask"), explicit=str(target))
    assert resolved == target.resolve()


def test_resolve_explicit_missing_dir_raises(tmp_path: Path):
    with pytest.raises(DatasetError, match="not a directory"):
        resolve_repo_root(_ds("flask"), explicit=str(tmp_path / "absent"))


def test_resolve_via_env_var(tmp_path: Path, monkeypatch):
    clone = tmp_path / "envflask"
    clone.mkdir()
    monkeypatch.setenv("OCE_BENCH_REPO_FLASK", str(clone))
    resolved = resolve_repo_root(_ds("flask"))
    assert resolved == clone.resolve()


def test_resolve_env_var_name_uppercases_hyphens(tmp_path: Path, monkeypatch):
    clone = tmp_path / "ccclone"
    clone.mkdir()
    # cc-switch -> OCE_BENCH_REPO_CC_SWITCH（连字符转下划线、大写）
    monkeypatch.setenv("OCE_BENCH_REPO_CC_SWITCH", str(clone))
    resolved = resolve_repo_root(_ds("cc-switch"))
    assert resolved == clone.resolve()


def test_resolve_env_var_missing_dir_raises(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OCE_BENCH_REPO_FLASK", str(tmp_path / "absent"))
    with pytest.raises(DatasetError, match="is not a directory"):
        resolve_repo_root(_ds("flask"))


def test_resolve_cwd_sibling(tmp_path: Path, monkeypatch):
    """cwd 下有同名目录 -> 命中（oce 仓库里跑时 flask 常是其同级）。"""
    monkeypatch.delenv("OCE_BENCH_REPO_FLASK", raising=False)
    workdir = tmp_path / "work"
    workdir.mkdir()
    sibling = workdir / "flask"
    sibling.mkdir()
    monkeypatch.chdir(workdir)
    resolved = resolve_repo_root(_ds("flask"))
    assert resolved == sibling.resolve()


def test_resolve_parent_sibling(tmp_path: Path, monkeypatch):
    """cwd 的父目录下有同名目录 -> 命中（在子目录里跑也能找到同级仓库）。"""
    monkeypatch.delenv("OCE_BENCH_REPO_FLASK", raising=False)
    parent = tmp_path / "parent"
    child = parent / "oce"
    child.mkdir(parents=True)
    sibling = parent / "flask"
    sibling.mkdir()
    monkeypatch.chdir(child)
    resolved = resolve_repo_root(_ds("flask"))
    assert resolved == sibling.resolve()


def test_resolve_no_fallback_raises_with_guidance(tmp_path: Path, monkeypatch):
    """全落空 -> 报错并指导用户传 --repo-root / 设环境变量；绝不回退到硬编码路径。"""
    monkeypatch.delenv("OCE_BENCH_REPO_FLASK", raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    with pytest.raises(DatasetError, match="cannot locate local clone") as ei:
        resolve_repo_root(_ds("flask", remote="https://example.com/flask.git"))
    msg = str(ei.value)
    assert "--repo-root" in msg
    assert "OCE_BENCH_REPO_FLASK" in msg
    assert "https://example.com/flask.git" in msg  # 给出 remote 便于 clone


def test_resolve_explicit_wins_over_env(tmp_path: Path, monkeypatch):
    """显式 --repo-root 优先于环境变量。"""
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    env_clone = tmp_path / "envclone"
    env_clone.mkdir()
    monkeypatch.setenv("OCE_BENCH_REPO_FLASK", str(env_clone))
    resolved = resolve_repo_root(_ds("flask"), explicit=str(explicit))
    assert resolved == explicit.resolve()


# ---------------------------------------------------------------------------
# 随包发布的真实数据集（Commit 8 迁入 src/oce/bench/datasets/ + package-data）
# ---------------------------------------------------------------------------


class TestShippedDatasets:
    """守卫随包数据集：迁入正确、可发现、metadata 配对、jsonl 合法。

    这组测的是**包里真实存在的文件**（非 tmp fixture），故能挡住"漏拷一份/metadata 配错/
    package-data glob 写错导致 wheel 里没有数据"这类回归。被测仓库是外部大仓、不进包，故此处
    只验数据集自身，不验 resolve_repo_root。
    """

    def test_two_datasets_shipped_and_discovered(self):
        from oce.bench.datasets import default_datasets_dir

        datasets = discover_datasets()  # 默认扫包内 datasets/
        aliases = {d.alias for d in datasets}
        assert "flask-retrieval-benchmark" in aliases
        assert "cc-switch-retrieval-benchmark" in aliases
        # 文件确实在包内目录（package-data 的来源）
        for d in datasets:
            assert d.queries_path.parent == default_datasets_dir()

    def test_each_dataset_has_paired_metadata(self):
        for d in discover_datasets():
            assert d.metadata_path is not None, f"{d.alias} 缺 metadata"
            assert d.metadata_path.is_file()
            assert d.questions == 100  # 两份都是 100 题

    def test_metadata_pins_repository_commit(self):
        """metadata 钉住被测仓库 commit/describe（RunRecord 溯源依赖）。"""
        flask = find_dataset("flask")
        assert flask.name == "flask"
        assert flask.repository.commit  # 非空
        assert flask.repository.describe  # tag/describe 非空
        assert "flask" in flask.repository.remote.lower()

    def test_jsonl_rows_are_valid_queries(self):
        """逐行校验 jsonl schema 完整（与 SKILL.md 第 7 节的自检口径一致）。"""
        for d in discover_datasets():
            with d.queries_path.open(encoding="utf-8-sig") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            assert len(rows) == d.questions
            ids = set()
            for row in rows:
                for key in ("category", "difficulty", "query", "expected_files"):
                    assert key in row, f"{d.alias}: 缺字段 {key}"
                qid = row.get("id") or row.get("query_id")
                assert qid and qid not in ids, f"{d.alias}: id 缺失或重复 {qid}"
                ids.add(qid)
                assert row["difficulty"] in (1, 2, 3)
                assert isinstance(row["expected_files"], list) and row["expected_files"]

    def test_find_by_short_name_resolves_shipped(self):
        assert find_dataset("flask").alias == "flask-retrieval-benchmark"
        assert find_dataset("cc-switch").alias == "cc-switch-retrieval-benchmark"
