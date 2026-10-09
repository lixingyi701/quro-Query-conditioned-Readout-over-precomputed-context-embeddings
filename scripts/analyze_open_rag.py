"""Strict same-question comparisons with paired CIs, sources and Hotpot hop types.

Positive delta means left minus right. Input files are src.train predictions.
Different evidence is permitted only for the explicit K5/K10 protocol diagnosis.
Training curves remain development evidence; this script does not pick a model
from benchmark test results or equate a single-seed CI with seed robustness.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import metrics


def read_predictions(path):
    rows = json.loads(Path(path).read_text())
    result = {str(row["id"]): row for row in rows}
    if not rows or len(result) != len(rows):
        raise ValueError(f"empty/duplicate prediction IDs: {path}")
    return result


def paired_interval(deltas, iterations, seed):
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(iterations))
    return [100 * samples[int(.025 * (iterations - 1))], 100 * samples[int(.975 * (iterations - 1))]]


def comparison(left, right, *, iterations=2000, seed=42, metadata=None, allow_evidence_change=False):
    if left.keys() != right.keys():
        raise ValueError("prediction IDs differ; intersection-only comparisons are forbidden")
    metadata = metadata or {}
    groups = {"all": []}
    for identifier in sorted(left):
        a, b = left[identifier], right[identifier]
        if a["query"] != b["query"] or a["golds"] != b["golds"]:
            raise ValueError(f"query/gold labels differ: {identifier}")
        if not allow_evidence_change:
            if "retrieved_doc_ids" not in a or "retrieved_doc_ids" not in b:
                raise ValueError("predictions lack evidence IDs; rerun eval with current code before a matched-evidence claim")
            if a["retrieved_doc_ids"] != b["retrieved_doc_ids"]:
                raise ValueError(f"ranked evidence differs: {identifier}")
        entry = (a, b)
        groups["all"].append(entry)
        info = metadata.get(identifier, a)
        for field in ("source", "hop_type"):
            if info.get(field):
                groups.setdefault(f"{field}:{info[field]}", []).append(entry)
    result = {}
    for name, pairs in groups.items():
        values = {"n": len(pairs)}
        for metric in ("substring", "em", "f1"):
            # Recompute every score from text under one implementation.
            a = [metrics.score(row["pred"], row["golds"])[metric] for row, _ in pairs]
            b = [metrics.score(row["pred"], row["golds"])[metric] for _, row in pairs]
            delta = [x - y for x, y in zip(a, b)]
            values[metric] = {"left_percent": 100 * sum(a) / len(a), "right_percent": 100 * sum(b) / len(b),
                              "delta_pp": 100 * sum(delta) / len(delta),
                              "ci95_pp": paired_interval(delta, iterations, seed)}
        result[name] = values
    return {"direction": "left_minus_right", "paired_by": "identical_question_ids_queries_gold_aliases",
            "evidence_change_allowed": allow_evidence_change, "groups": result,
            "uncertainty": "question bootstrap only; single seed does not measure training randomness"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--queries", help="optional original JSONL for source/hop grouping")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow_evidence_change", action="store_true", help="only for explicit evidence-protocol diagnosis")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    metadata = None
    if args.queries:
        with open(args.queries, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        metadata = {str(row["id"]): row for row in rows}
    result = comparison(read_predictions(args.left), read_predictions(args.right),
                        iterations=args.iterations, seed=args.seed, metadata=metadata,
                        allow_evidence_change=args.allow_evidence_change)
    out = Path(args.out)
    if out.exists():
        raise ValueError("analysis output already exists; choose a new path")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
