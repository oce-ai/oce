"""检索质量评分口径（纯函数，零 I/O）。

两个相互独立的轴，每题各 1 分、共 2 分：

- **Top-1** —— 排名第一的结果是否答对了问题；排序栈的天花板，故意严苛（1 分，硬判）。
- **nDCG@10** —— 整个返回窗口有多可用；位置感知的部分得分，能识别"更靠后才找到支撑文件"
  （0..1 分）。

``expected_files`` 按"最相关在前"编写：首项是真正回答问题的文件（rel=2），其余是支撑上下文
（rel=1）。这是唯一可用的排序信号——没有它，nDCG 无法区分"找到了定义"和"找到了调用方"。

从 oce-benchmark/scripts/run_retrieval_eval.py 逐字节等价移植：评分是历史分数可比性的基础，
改一个常数就会让所有旧报告作废。新增测试 test_scoring.py 锁定回归。
"""

from __future__ import annotations

import math
from fnmatch import fnmatch

# 评分参数：Top-1（1 分）+ nDCG@10（0..1 分），每题 2 分。
NDCG_K = 10
POINTS_PER_QUERY = 2
# expected_files 首项 = 真正的答案（rel=2），其余 = 支撑上下文（rel=1）。
PRIMARY_GRADE = 2
SUPPORTING_GRADE = 1


def extract_paths(formatted: str) -> list[str]:
    """从 ``formatted_retrieval`` 文本里按出现顺序抽出命中路径（去重）。

    评测消费的是服务返回的**人读格式**（``Path: xxx`` 行），而非结构化 hits —— 这与真实
    客户端看到的内容一致，故评分反映的是端到端可用结果，而非内部中间态。
    """
    paths: list[str] = []
    seen: set[str] = set()
    for line in formatted.splitlines():
        if not line.startswith("Path: "):
            continue
        path = line[6:].strip()
        if path not in seen:
            seen.add(path)
            paths.append(path)
    return paths


def path_matches(actual: str, expected: str) -> bool:
    """比对一个返回路径与一条 expected 项，支持 glob。

    数据集可表达一族文件（如 ``src-tauri/src/commands/*.rs``），纯等值比对会让这些行
    无法评分。仅当 expected 含 glob 元字符（``*?[``）时走 fnmatch，否则精确等值——
    避免把含 ``[`` 的普通路径误当字符类。
    """
    if any(ch in expected for ch in "*?["):
        return fnmatch(actual, expected)
    return actual == expected


def relevance_grades(expected: list[str]) -> list[int]:
    """按位置给 expected 文件打分：首个（答案）高于其余（支撑）。"""
    return [
        PRIMARY_GRADE if index == 0 else SUPPORTING_GRADE
        for index in range(len(expected))
    ]


def dcg(gains: list[int]) -> float:
    """Discounted Cumulative Gain：位置越靠后，增益折扣越大（log2 折减）。"""
    return sum(
        gain / math.log2(rank + 1)
        for rank, gain in enumerate(gains, 1)
        if gain
    )


def ndcg_at_k(top_paths: list[str], expected: list[str], k: int = NDCG_K) -> float:
    """前 k 个返回路径上的归一化 DCG。

    每条 expected 至多被消费一次，故重复返回同一文件无法刷分。分母是"理想排序"（所有
    expected 按 grade 降序排在最前）的 DCG，使结果落在 [0, 1]。
    """
    grades = relevance_grades(expected)
    remaining = set(range(len(expected)))
    gains: list[int] = []
    for actual in top_paths[:k]:
        matched_index = next(
            (index for index in remaining if path_matches(actual, expected[index])),
            None,
        )
        if matched_index is None:
            gains.append(0)
            continue
        remaining.discard(matched_index)
        gains.append(grades[matched_index])

    ideal = sorted(grades, reverse=True)[:k]
    denominator = dcg(ideal)
    if not denominator:
        return 0.0
    return dcg(gains) / denominator


def score_query(query: dict, formatted: str) -> tuple[list[str], int, float]:
    """对单个查询在两个独立轴上打分，各 1 分。

    Returns:
        ``(top_paths, top1_score, ndcg_score)``：
        - ``top_paths``    返回窗口里的命中路径（去重、按序）
        - ``top1_score``   首位是否命中任一 expected（0/1，硬判）
        - ``ndcg_score``   整个 top-10 窗口的可用性（0..1，位置感知部分分）

    ``expected_files`` 里的反斜杠归一为正斜杠，使 Windows 路径与数据集一致。
    """
    expected = [path.replace("\\", "/") for path in query["expected_files"]]
    top_paths = extract_paths(formatted)
    top1 = top_paths[0] if top_paths else ""
    top1_score = (
        int(any(path_matches(top1, item) for item in expected)) if top1 else 0
    )
    return top_paths, top1_score, ndcg_at_k(top_paths, expected)
