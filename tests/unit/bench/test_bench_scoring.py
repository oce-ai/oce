"""评分口径回归测试。

这些期望值已用一次性对拍脚本确认与 oce-benchmark/scripts/run_retrieval_eval.py **逐字节
等价**（35 组 formatted × expected 组合全等）。固化成自包含断言，锁死评分口径——它是历史
分数可比性的基础，改一个常数就会让所有旧报告作废。
"""

from __future__ import annotations

import pytest

from oce.bench.scoring import (
    NDCG_K,
    POINTS_PER_QUERY,
    PRIMARY_GRADE,
    SUPPORTING_GRADE,
    dcg,
    extract_paths,
    ndcg_at_k,
    path_matches,
    relevance_grades,
    score_query,
)


def test_constants():
    assert NDCG_K == 10
    assert POINTS_PER_QUERY == 2
    assert PRIMARY_GRADE == 2
    assert SUPPORTING_GRADE == 1


class TestExtractPaths:
    def test_extracts_in_order(self):
        formatted = "Path: a.py\nsnippet\nPath: b/c.rs\nmore"
        assert extract_paths(formatted) == ["a.py", "b/c.rs"]

    def test_dedupes_repeats(self):
        formatted = "Path: a.py\nPath: a.py\nPath: b.rs"
        assert extract_paths(formatted) == ["a.py", "b.rs"]

    def test_empty_when_no_path_lines(self):
        assert extract_paths("") == []
        assert extract_paths("no paths here\nrandom text") == []

    def test_ignores_non_path_prefix(self):
        # 只有 "Path: " 开头的行算命中；相似前缀不算
        assert extract_paths("PathX: a.py\nPath: b.py") == ["b.py"]


class TestPathMatches:
    def test_exact_equality(self):
        assert path_matches("a.py", "a.py") is True
        assert path_matches("a.py", "b.py") is False

    def test_glob_star(self):
        assert path_matches("a.py", "*.py") is True
        assert path_matches("src/commands/foo.rs", "src/commands/*.rs") is True

    def test_glob_question_and_class(self):
        assert path_matches("a1.py", "a?.py") is True
        assert path_matches("ab.py", "a[bc].py") is True

    def test_bracket_in_literal_path_is_treated_as_glob(self):
        """含 [ 的 expected 一律走 fnmatch（已知取舍）：字面 'x[1].py' 当字符类解析，
        故不匹配字面串本身。移植时保留此行为以与原 harness 逐字节一致。"""
        assert path_matches("x[1].py", "x[1].py") is False
        # 但它会匹配 x1.py / x.py 之外的字符类展开
        assert path_matches("x1.py", "x[1].py") is True


class TestRelevanceGrades:
    def test_primary_first_rest_supporting(self):
        assert relevance_grades(["a", "b", "c"]) == [2, 1, 1]

    def test_single(self):
        assert relevance_grades(["a"]) == [2]

    def test_empty(self):
        assert relevance_grades([]) == []


class TestDcg:
    def test_position_discount(self):
        # gain 2 at rank1 -> 2/log2(2)=2; gain 1 at rank2 -> 1/log2(3)
        assert dcg([2, 1]) == pytest.approx(2 + 1 / 1.5849625007211563)

    def test_zero_gains_skipped(self):
        assert dcg([0, 0]) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking_scores_one(self):
        assert ndcg_at_k(["a.py"], ["a.py"]) == pytest.approx(1.0)

    def test_no_match_scores_zero(self):
        assert ndcg_at_k(["zzz.py"], ["a.py"]) == 0.0

    def test_empty_expected_scores_zero(self):
        # 无 expected -> ideal DCG=0 -> 定义返回 0.0
        assert ndcg_at_k(["a.py"], []) == 0.0

    def test_position_penalty_when_order_flipped(self):
        """找到全部但顺序反了：支撑文件在前、答案在后 -> 低于 1.0。"""
        correct = ndcg_at_k(["a.py", "b.py"], ["a.py", "b.py"])
        flipped = ndcg_at_k(["b.py", "a.py"], ["a.py", "b.py"])
        assert correct == pytest.approx(1.0)
        assert flipped < correct
        assert flipped == pytest.approx(0.8597, abs=1e-3)

    def test_partial_credit_for_finding_support_only(self):
        # 只找到支撑文件（rel=1），未找到答案（rel=2）
        score = ndcg_at_k(["b.py"], ["a.py", "b.py"])
        assert 0.0 < score < 1.0

    def test_repeat_same_file_cannot_inflate(self):
        """每条 expected 至多消费一次：重复返回同一文件不刷分。"""
        once = ndcg_at_k(["a.py"], ["a.py", "b.py"])
        repeated = ndcg_at_k(["a.py", "a.py"], ["a.py", "b.py"])
        assert repeated == pytest.approx(once)

    def test_k_truncates_window(self):
        paths = ["a.py"] + [f"noise{i}.py" for i in range(20)]
        assert ndcg_at_k(paths, ["a.py"], k=1) == pytest.approx(1.0)
        # a.py 排在第 21 位，k=10 窗口看不到 -> 0
        late = [f"noise{i}.py" for i in range(20)] + ["a.py"]
        assert ndcg_at_k(late, ["a.py"], k=10) == 0.0


class TestScoreQuery:
    def test_top1_hit_and_ndcg(self):
        query = {"expected_files": ["a.py", "b.py"]}
        formatted = "Path: a.py\nPath: b.py"
        top_paths, top1, ndcg = score_query(query, formatted)
        assert top_paths == ["a.py", "b.py"]
        assert top1 == 1
        assert ndcg == pytest.approx(1.0)

    def test_top1_miss_but_ndcg_partial(self):
        """首位错、但答案在窗口内：top1=0，ndcg>0（两轴独立）。"""
        query = {"expected_files": ["a.py"]}
        formatted = "Path: noise.py\nPath: a.py"
        top_paths, top1, ndcg = score_query(query, formatted)
        assert top1 == 0
        assert ndcg > 0.0

    def test_empty_result_scores_zero(self):
        query = {"expected_files": ["a.py"]}
        top_paths, top1, ndcg = score_query(query, "")
        assert top_paths == []
        assert top1 == 0
        assert ndcg == 0.0

    def test_backslash_normalized(self):
        """expected_files 里的 Windows 反斜杠归一为正斜杠再比对。"""
        query = {"expected_files": ["src\\flask\\app.py"]}
        formatted = "Path: src/flask/app.py"
        _, top1, ndcg = score_query(query, formatted)
        assert top1 == 1
        assert ndcg == pytest.approx(1.0)

    def test_glob_expected_scoreable(self):
        query = {"expected_files": ["src/commands/*.rs"]}
        formatted = "Path: src/commands/foo.rs"
        _, top1, ndcg = score_query(query, formatted)
        assert top1 == 1
        assert ndcg == pytest.approx(1.0)
