"""Evaluate the production hybrid intent classifier on a JSONL regression set.

The script is deliberately independent of the retrieval service and makes no
network calls in its default rules-only mode.  It records the resolver source
and reason so a 100% score cannot be mistaken for a neural-model score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from oce.domain.services.llm.intent import HybridIntentClassifier


LABELS = ("S", "C", "R", "P", "F", "O", "M")


async def evaluate(data: Path, *, legacy_benchmark: bool = False) -> dict:
    rows = [
        json.loads(line)
        for line in data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # The frozen 84-case set predates hybrid-v2 and labels static-reference
    # lookups as S.  Keep that convention available only as an explicit
    # compatibility mode; the default evaluates the production resolver.
    classifier = HybridIntentClassifier(
        model="rules-only",
        soft_provider=None,
        resolver_version=("legacy-benchmark-v1" if legacy_benchmark else "hybrid-v2"),
        legacy_static_reference_as_symbol=legacy_benchmark,
    )
    confusion = {label: Counter() for label in LABELS}
    sources = Counter()
    reasons = Counter()
    latencies: list[float] = []
    errors: list[dict] = []
    for row in rows:
        started = perf_counter()
        predicted, decision = await classifier.classify_with_decision(str(row["query"]))
        latencies.append((perf_counter() - started) * 1000)
        gold = str(row.get("label", row.get("gold_intent", ""))).upper()
        got = predicted.value
        confusion.setdefault(gold, Counter())[got] += 1
        sources[decision.source] += 1
        reasons[decision.reason] += 1
        if got != gold:
            errors.append(
                {
                    "id": row.get("id"),
                    "gold": gold,
                    "predicted": got,
                    "reason": decision.reason,
                    "query": row.get("query", ""),
                }
            )

    total = len(rows)
    correct = total - len(errors)
    per_class = {}
    for label in LABELS:
        support = sum(confusion.get(label, {}).values())
        tp = confusion.get(label, {}).get(label, 0)
        predicted_count = sum(row.get(label, 0) for row in confusion.values())
        recall = tp / support if support else 0.0
        precision = tp / predicted_count if predicted_count else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        per_class[label] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(LABELS)
    ordered = sorted(latencies)

    def percentile(q: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, int((len(ordered) - 1) * q))
        return ordered[index]

    return {
        "resolver": "oce.HybridIntentClassifier",
        "mode": "legacy-benchmark-rules-only" if legacy_benchmark else "hybrid-v2-rules-only",
        "data": str(data),
        "n": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": {label: dict(confusion.get(label, {})) for label in LABELS},
        "source_counts": dict(sources),
        "decision_reasons": dict(reasons),
        "latency_p50_ms": percentile(0.50),
        "latency_p95_ms": percentile(0.95),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--legacy-benchmark",
        action="store_true",
        help="reproduce the frozen 84-case S-label convention; not a production resolver",
    )
    args = parser.parse_args()
    report = asyncio.run(evaluate(args.data, legacy_benchmark=args.legacy_benchmark))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
