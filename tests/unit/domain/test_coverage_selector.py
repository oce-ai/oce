from __future__ import annotations

from oce.domain.services.search import SearchHit
from oce.domain.services.selector.coverage_selector import CoverageSelector


def _hit(path: str, start: int, end: int, score: float, content: str = "code") -> SearchHit:
    return SearchHit(
        blob_name=path,
        path=path,
        content=content,
        score=score,
        start_line=start,
        end_line=end,
    )


async def test_prefers_cross_file_coverage_before_second_chunk():
    selector = CoverageSelector(max_per_path=2, max_chars=10_000)
    hits = [
        _hit("src/a.py", 1, 10, 0.9),
        _hit("src/a.py", 20, 30, 0.8),
        _hit("src/b.py", 1, 10, 0.7),
    ]

    selected = await selector.select(hits, 3)

    assert [(hit.path, hit.start_line) for hit in selected] == [
        ("src/a.py", 1),
        ("src/b.py", 1),
        ("src/a.py", 20),
    ]


async def test_suppresses_highly_overlapping_spans():
    selector = CoverageSelector(overlap_threshold=0.5)
    hits = [
        _hit("src/a.py", 1, 10, 0.9),
        _hit("src/a.py", 5, 12, 0.8),
        _hit("src/a.py", 20, 25, 0.7),
    ]

    selected = await selector.select(hits, 3)

    assert [(hit.start_line, hit.end_line) for hit in selected] == [
        (1, 10),
        (20, 25),
    ]


async def test_enforces_character_budget_but_keeps_best_hit():
    selector = CoverageSelector(max_chars=5)
    hits = [
        _hit("src/a.py", 1, 1, 0.9, content="longer than budget"),
        _hit("src/b.py", 1, 1, 0.8, content="also long"),
    ]

    selected = await selector.select(hits, 2)

    assert [hit.path for hit in selected] == ["src/a.py"]
