from __future__ import annotations

import pytest

from oce.domain.services.query_planner import HeuristicQueryPlanner


def test_keeps_short_or_single_facet_query_unchanged():
    planner = HeuristicQueryPlanner()

    assert planner.plan("Where is authentication implemented?") == [
        "Where is authentication implemented?"
    ]


def test_keeps_complete_request_and_extracts_explicit_facets():
    planner = HeuristicQueryPlanner(max_queries=3)

    assert planner.plan(
        "Find request authentication. Trace credential reload. Explain error mapping."
    ) == [
        "Find request authentication. Trace credential reload. Explain error mapping.",
        "Find request authentication",
        "Trace credential reload",
    ]


def test_does_not_split_file_extensions():
    planner = HeuristicQueryPlanner()

    assert planner.plan("Explain src/oce/main.py and its startup lifecycle") == [
        "Explain src/oce/main.py and its startup lifecycle"
    ]


def test_validates_limits():
    with pytest.raises(ValueError):
        HeuristicQueryPlanner(max_queries=0)
    with pytest.raises(ValueError):
        HeuristicQueryPlanner(min_facet_chars=0)
