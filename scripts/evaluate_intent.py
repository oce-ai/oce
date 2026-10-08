"""在 OCE 自有回归集上评测意图判定表。

默认纯规则模式：不加载任何模型、不发起任何网络请求，因此可在离线环境跑。
报告含 accuracy、macro F1、每类 P/R/F1、混淆矩阵、判定来源分布、reason
分布与边界对统计。

用法::

    python scripts/evaluate_intent.py --data tests/data/intent-benchmark.jsonl
    python scripts/evaluate_intent.py --data ... --out bench/runs/intent.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

from oce.domain.services.intent.resolver import resolve_rules
from oce.domain.services.intent.taxonomy import BOUNDARY_PAIRS, LABELS, intent_from_label


def _percentile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, int((len(ordered) - 1) * q))
    return ordered[index]


def evaluate(data: Path) -> dict:
    rows = [
        json.loads(line)
        for line in data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    codes = [label.value for label in LABELS]
    confusion: dict[str, Counter[str]] = {code: Counter() for code in codes}
    sources: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    latencies: list[float] = []
    errors: list[dict] = []
    skipped: list[dict] = []

    for row in rows:
        query = str(row.get("query", ""))
        gold = intent_from_label(str(row.get("label", row.get("gold_intent", ""))))
        if not query or gold is None:
            skipped.append({"id": row.get("id"), "reason": "missing_query_or_label"})
            continue

        started = perf_counter()
        decision = resolve_rules(query)
        latencies.append((perf_counter() - started) * 1000)

        predicted = decision.intent
        confusion[gold.value][predicted.value] += 1
        sources[decision.source] += 1
        reasons[decision.reason] += 1
        if predicted is not gold:
            errors.append(
                {
                    "id": row.get("id"),
                    "gold": gold.value,
                    "predicted": predicted.value,
                    "reason": decision.reason,
                    "query": query,
                }
            )

    total = sum(sum(counter.values()) for counter in confusion.values())
    correct = total - len(errors)

    per_class: dict[str, dict] = {}
    for code in codes:
        support = sum(confusion[code].values())
        tp = confusion[code][code]
        predicted_count = sum(confusion[other][code] for other in codes)
        recall = tp / support if support else 0.0
        precision = tp / predicted_count if predicted_count else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        per_class[code] = {
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    macro_f1 = sum(item["f1"] for item in per_class.values()) / len(codes)

    boundary = {
        f"{left.value}<->{right.value}": {
            "a_as_b": confusion[left.value][right.value],
            "b_as_a": confusion[right.value][left.value],
        }
        for left, right in BOUNDARY_PAIRS
    }

    ordered = sorted(latencies)
    return {
        "resolver": "oce.domain.services.intent.resolve_rules",
        "mode": "rules-only",
        "data": str(data),
        "n": total,
        "skipped": skipped,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion_matrix": {code: dict(confusion[code]) for code in codes},
        "boundary": boundary,
        "source_counts": dict(sources),
        "decision_reasons": dict(reasons),
        "latency_p50_ms": _percentile(ordered, 0.50),
        "latency_p95_ms": _percentile(ordered, 0.95),
        "errors": errors,
    }


def _print_summary(report: dict) -> None:
    print("\n================ INTENT EVALUATION ================")
    print(f"data            : {report['data']}")
    print(f"evaluated       : {report['n']}  (skipped {len(report['skipped'])})")
    print(f"accuracy        : {report['accuracy']:.4f}")
    print(f"macro F1        : {report['macro_f1']:.4f}")
    print(f"latency p50/p95 : {report['latency_p50_ms']:.3f} / {report['latency_p95_ms']:.3f} ms")

    print("\nper-class:")
    print(f"  {'lab':>3} {'support':>7} {'recall':>7} {'prec':>7} {'f1':>7}")
    for code, item in report["per_class"].items():
        print(
            f"  {code:>3} {item['support']:>7} {item['recall']:>7.3f} "
            f"{item['precision']:>7.3f} {item['f1']:>7.3f}"
        )

    print("\ndecision sources:")
    for source, count in sorted(report["source_counts"].items()):
        print(f"  {source}: {count}")

    if report["errors"]:
        print(f"\nerrors ({len(report['errors'])}):")
        for item in report["errors"]:
            print(
                f"  {item['id']}: gold={item['gold']} pred={item['predicted']} "
                f"reason={item['reason']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("tests/data/intent-benchmark.jsonl"),
        help="JSONL with query + label fields",
    )
    parser.add_argument("--out", type=Path, help="optional JSON report path")
    args = parser.parse_args()

    report = evaluate(args.data)
    _print_summary(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nreport -> {args.out}")


if __name__ == "__main__":
    main()
