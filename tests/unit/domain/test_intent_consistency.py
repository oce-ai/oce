"""跨实现一致性与解耦守卫。

重构前三条判定路径对同一查询会给出不同标签（16 条抽样里三方不一致 5 条）。
这里断言现在只剩一条真源：同步入口、纯规则入口、异步判定器的降级路径在整个
回归集上必须给出完全相同的标签。

另外守住与 oce-laya 的解耦：源码/测试/脚本/配置里不得再出现对它的引用。
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import re

import pytest

from oce.domain.services.intent.resolver import IntentResolver, resolve_rules
from oce.domain.services.intent.taxonomy import LABELS, QueryIntent, intent_from_label
from oce.domain.services.query_classifier import (
    classify_query_intent,
    is_filename_query,
    should_use_path_index,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DATASET = REPO_ROOT / "tests" / "data" / "intent-benchmark.jsonl"


def _dataset_rows() -> list[dict]:
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_dataset_exists_and_covers_every_label() -> None:
    rows = _dataset_rows()
    assert rows, "回归集不能为空"
    labels = {row["label"] for row in rows}
    assert labels == {label.value for label in LABELS}, (
        f"回归集未覆盖全部标签，缺少: {{l.value for l in LABELS}} - {labels}"
    )


def test_dataset_labels_are_all_valid() -> None:
    for row in _dataset_rows():
        assert intent_from_label(row["label"]) is not None, f"非法标签: {row}"


def test_required_divergence_cases_are_present_with_adjudicated_labels() -> None:
    """契约点名要求覆盖的分歧用例，及其裁定标签。"""
    rows = {row["query"]: row["label"] for row in _dataset_rows()}
    expected = {
        "trace the call chain from the CLI entry to the index writer": "C",
        "在 src/oce/domain 目录下哪个文件实现了分块逻辑？": "F",
        "How do I call `build_index` and what does it return?": "R",
        "哪些文件引用了 `parse_config`？": "U",
        "找出 `parse_config` 的定义，并说明它的调用链路": "M",
    }
    for query, label in expected.items():
        assert query in rows, f"回归集缺少分歧用例: {query}"
        assert rows[query] == label, f"{query} 的 gold 应为 {label}，实际 {rows[query]}"


def test_sync_entry_and_rules_entry_never_disagree() -> None:
    """同步兼容入口与纯规则入口必须完全一致。"""
    mismatches = []
    for row in _dataset_rows():
        query = row["query"]
        via_sync = classify_query_intent(query)
        via_rules = resolve_rules(query).intent
        if via_sync is not via_rules:
            mismatches.append((row.get("id"), via_sync.value, via_rules.value))
    assert not mismatches, f"两条入口不一致: {mismatches}"


def test_resolver_fallback_matches_the_rule_table() -> None:
    """判定源缺席时，异步判定器必须与纯规则入口逐条一致。"""
    resolver = IntentResolver(provider=None)

    async def run() -> list[tuple]:
        mismatches = []
        for row in _dataset_rows():
            query = row["query"]
            decision = await resolver.resolve(query)
            expected = resolve_rules(query).intent
            if decision.intent is not expected:
                mismatches.append((row.get("id"), decision.intent.value, expected.value))
        return mismatches

    assert not asyncio.run(run())


def test_rules_are_deterministic() -> None:
    """同一查询重复判定必须稳定（无隐藏状态）。"""
    for row in _dataset_rows()[:30]:
        query = row["query"]
        results = {resolve_rules(query).intent for _ in range(5)}
        assert len(results) == 1, f"判定不稳定: {query} -> {results}"


def test_compat_entries_agree_with_the_taxonomy() -> None:
    """保留的兼容入口必须与意图判定同源。"""
    for row in _dataset_rows():
        query = row["query"]
        is_path = resolve_rules(query).intent is QueryIntent.PATH
        assert should_use_path_index(query) is is_path
        assert is_filename_query(query)[0] is is_path


# ── 解耦守卫 ────────────────────────────────────────────────────────────────

SCAN_DIRS = ("src", "tests", "scripts", "bench", "docs")
FORBIDDEN = re.compile(r"oce-laya|hybrid_intent|hybrid_predictor|\blaya\b", re.IGNORECASE)
# 契约允许在说明性文字里提及历史实现，禁止的是代码/路径/配置引用。因此扫描前
# 先剥掉 Python 的注释与 docstring，只看真正会被执行或被解析的内容。
DOC_SUFFIXES = {".md"}
CODE_SUFFIXES = {".py", ".toml", ".json", ".jsonl", ".yml", ".yaml", ".cfg", ".ini"}


def _strip_comments_and_docstrings(source: str) -> str:
    """去掉注释与 docstring，保留其余代码（含普通字符串字面量）。"""
    import ast
    import io as _io
    import tokenize

    # 1. 收集所有 docstring 的字面量文本，稍后整体剔除。
    docstrings: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    docstrings.append(doc)

    # 2. 去掉注释 token。
    pieces: list[str] = []
    try:
        for token in tokenize.generate_tokens(_io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                continue
            pieces.append(token.string)
    except (tokenize.TokenError, IndentationError):
        pieces = [source]
    code = "\n".join(pieces)

    # 3. 剔除 docstring 内容。
    for doc in docstrings:
        code = code.replace(doc, " ")
    return code


def test_no_reference_to_the_oce_laya_project() -> None:
    offenders: list[str] = []
    for directory in SCAN_DIRS:
        base = REPO_ROOT / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix in DOC_SUFFIXES:
                continue
            if path.suffix not in CODE_SUFFIXES or "__pycache__" in path.parts:
                continue
            # 本守卫自身的 FORBIDDEN 正则字面量必然命中，跳过自己。
            if path.resolve() == pathlib.Path(__file__).resolve():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if path.suffix == ".py":
                text = _strip_comments_and_docstrings(text)
            if FORBIDDEN.search(text):
                hits = sorted(set(FORBIDDEN.findall(text)))
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {hits}")
    assert not offenders, "仍存在对 oce-laya 的代码级引用:\n" + "\n".join(offenders)


def test_only_one_query_intent_enum_in_src() -> None:
    """src/ 下只允许存在一个意图枚举定义。"""
    definitions = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^class QueryIntent\b", text, re.MULTILINE):
            definitions.append(str(path.relative_to(REPO_ROOT)))
    assert definitions == ["src/oce/domain/services/intent/taxonomy.py".replace("/", "\\")] or \
           definitions == ["src/oce/domain/services/intent/taxonomy.py"], (
        f"意图枚举定义应唯一，实际: {definitions}"
    )


def test_no_removed_compatibility_symbols_remain() -> None:
    """已删除的兼容层与死代码不得残留在 src/。"""
    removed = (
        "legacy_static_reference_as_symbol",
        "conservative_rule",
        "ShadowIntentClassifier",
        "HybridIntentClassifier",
        "HeuristicIntentClassifier",
        "LayaSoftSignalProvider",
        "SoftSignals",
        "HardSignals",
        "_HARD_TO_LLM",
        "neural_first",
        "rule_first",
    )
    offenders: list[str] = []
    for path in (REPO_ROOT / "src").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for name in removed:
            if name in text:
                offenders.append(f"{path.relative_to(REPO_ROOT)}: {name}")
    assert not offenders, f"残留的已删除符号: {offenders}"


def _signals_field_names() -> set[str]:
    """读取 Signals 数据类实际声明的字段名（不看注释与 docstring）。"""
    from oce.domain.services.intent.signals import Signals

    return set(Signals.__dataclass_fields__)


def test_no_duplicate_signal_field_names() -> None:
    """不得同时存在同义异名的信号字段。

    断言对象是数据类真实声明的字段，而不是文件里的任意文本；说明性文字提到
    旧字段名是允许的。
    """
    fields = _signals_field_names()
    banned = {
        "has_flow_delimiter",
        "has_implementation_marker",
        "has_api_usage_marker",
        "has_overview_marker",
        "has_definition_marker",
        "has_reference_marker",
        "independent_clauses",
        "has_concrete_symbol",
        "symbol_tokens",
        "has_extension",
        "identifier_count",
    }
    leftover = fields & banned
    assert not leftover, f"同义异名字段仍存在: {sorted(leftover)}"


def test_each_semantic_signal_has_exactly_one_field() -> None:
    """每个语义信号只有一个字段名。"""
    fields = _signals_field_names()
    for expected in (
        "has_flow",
        "has_reference",
        "has_usage",
        "has_definition",
        "has_overview",
        "has_feature",
        "has_path",
        "has_symbol",
        "has_independent_clauses",
    ):
        assert expected in fields, f"缺少规范字段: {expected}"


@pytest.mark.parametrize(
    "module",
    [
        "oce.domain.services.intent_resolver",
        "oce.domain.services.llm.intent",
    ],
)
def test_removed_modules_are_gone(module: str) -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)
