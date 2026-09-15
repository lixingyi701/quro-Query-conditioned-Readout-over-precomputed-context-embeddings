"""Re-score every arm on the questions that actually require the retrieved evidence.

A 7B decoder answers a large share of TriviaQA closed-book.  Measured here, the
same model scores 64.1% EM when handed *another question's* documents versus
67.9% with the right ones: the whole retrieval-and-compression pipeline is worth
3.8 points, and every difference between readouts lives inside that band.  Arm
comparisons on the full set are therefore mostly a measurement of Mistral's
parametric memory.

This script defines the evidence-dependent subset as the questions the model gets
*wrong* when given mismatched documents, then re-scores every arm on exactly those
questions.  On that subset a correct answer cannot come from parametric recall, so
the numbers isolate the channel the method is about.

It is pure re-analysis of saved predictions -- no GPU, no retraining.

    python scripts/evidence_subset.py --runs /data02/quro/runs \
      --control gonogo_C_doccontrol --split trivia --budget 8
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics, paths


def load_predictions(run_dir: str, split: str, mode: str, budget: int):
    path = os.path.join(run_dir, f"predictions_{split}_{mode}_B{budget}.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return {row["id"]: row for row in json.load(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=paths.RUNS_DIR)
    ap.add_argument("--control", required=True,
                    help="run directory containing the mismatch-doc predictions")
    ap.add_argument("--split", default="trivia")
    ap.add_argument("--mode", default="D0")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--metric", default="em", choices=["em", "f1", "substring"])
    args = ap.parse_args()

    control_dir = os.path.join(args.runs, args.control)
    mismatched = load_predictions(control_dir, f"{args.split}_mismatch-doc",
                                  args.mode, args.budget)
    matched = load_predictions(control_dir, args.split, args.mode, args.budget)
    if not mismatched or not matched:
        raise SystemExit(f"{control_dir} has no mismatch-doc predictions for "
                         f"{args.split}/{args.mode}/B={args.budget}; rerun with --doc_control")

    # Answerable without the right evidence -> excluded.
    evidence_needed = {i for i, row in mismatched.items() if row[args.metric] < 1.0}
    total = len(mismatched)
    print(f"split={args.split} mode={args.mode} B={args.budget} metric={args.metric}")
    print(f"  {total} scored rows; the decoder answers {total - len(evidence_needed)} "
          f"({(total - len(evidence_needed)) / total:.1%}) with mismatched documents")
    print(f"  evidence-dependent subset: {len(evidence_needed)} rows "
          f"({len(evidence_needed) / total:.1%})\n")

    header = f"{'run':<26} {'full set':>10} {'evidence subset':>17} {'n':>6}"
    print(header)
    print("-" * len(header))
    rows = []
    for run_dir in sorted(glob.glob(os.path.join(args.runs, "*"))):
        if not os.path.isdir(run_dir):
            continue
        predictions = load_predictions(run_dir, args.split, args.mode, args.budget)
        if not predictions:
            continue
        shared = [p for i, p in predictions.items() if i in mismatched]
        subset = [p for i, p in predictions.items() if i in evidence_needed]
        if not shared or not subset:
            continue
        full_score = sum(p[args.metric] for p in shared) / len(shared)
        subset_score = sum(p[args.metric] for p in subset) / len(subset)
        rows.append((os.path.basename(run_dir), full_score, subset_score, len(subset)))
        print(f"{os.path.basename(run_dir):<26} {full_score:>9.2%} {subset_score:>16.2%} "
              f"{len(subset):>6}")

    if len(rows) > 1:
        best_full = max(rows, key=lambda r: r[1])
        best_subset = max(rows, key=lambda r: r[2])
        print(f"\n  best on the full set        : {best_full[0]} ({best_full[1]:.2%})")
        print(f"  best on the evidence subset : {best_subset[0]} ({best_subset[2]:.2%})")
        if best_full[0] != best_subset[0]:
            print("  the ranking changes once parametric recall is removed -- "
                  "report the subset result")


if __name__ == "__main__":
    main()
