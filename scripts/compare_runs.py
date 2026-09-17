"""Compare finished runs against one baseline, with paired tests and the floor.

``collect_arm_matrix.py`` maintains the project-wide table from a fixed registry;
this is for the question asked right after a wave finishes -- did these arms beat
that one -- without editing a registry first.

Three things it refuses to let a reader skip:

*The floor.* Accuracy is printed next to its mismatch-document control and the
evidence value is the difference. On TriviaQA a 7B answers 59% with the wrong
documents in front of it, so a bare accuracy number cannot be read at all.

*Both metrics.* EM and substring are printed side by side. When they disagree in
direction the result is reported as such, not collapsed into whichever one moved.

*The validation curve.* The final step is not necessarily the best step, and a run
that peaked early and decayed looks identical to one that never got there if only
the endpoint is shown.

    python scripts/compare_runs.py --baseline r2a_C1 --runs r2b_C1 r2c_C1 r2d_S
"""

from __future__ import annotations

import argparse
import json
import os
from math import comb


def load_result(runs_dir, tag):
    path = os.path.join(runs_dir, tag, "result.json")
    return json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None


def load_predictions(runs_dir, tag, split, mode, budget):
    path = os.path.join(runs_dir, tag, f"predictions_{split}_{mode}_B{budget}.json")
    if not os.path.exists(path):
        return None
    return {row["id"]: row for row in json.load(open(path, encoding="utf-8"))}


def mcnemar(deltas):
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1))
    return wins, losses, min(1.0, 2 * tail / 2 ** n)


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def validation_curve(runs_dir, tag):
    path = os.path.join(runs_dir, tag, "train_log.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "validation" in record:
                out.append((record["validation"]["step"], record["validation"]["em"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="/data02/quro/runs")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", default="D0")
    # Runs that differ in what the decoder reads are still compared per question:
    # the arms answer the same 2000 items, so the pairing is by id and the modes
    # only decide which metric key and which predictions file to read.
    ap.add_argument("--baseline_mode", default=None,
                    help="decoder input mode of the baseline, if it differs")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--label", nargs="*", default=None,
                    help="human-readable name per run, baseline first")
    args = ap.parse_args()

    tags = [args.baseline] + args.runs
    labels = dict(zip(tags, args.label)) if args.label else {}
    baseline_mode = args.baseline_mode or args.mode
    modes = {tag: (baseline_mode if tag == args.baseline else args.mode) for tag in tags}

    print(f"{'run':<10}{'label':<20}{'EM':>8}{'sub':>8}{'floor':>8}{'evidence':>10}"
          f"{'best val':>10}")
    print("-" * 74)
    available = []
    for tag in tags:
        clean = f"{args.split}|{modes[tag]}|B={args.budget}"
        floor = f"{args.split}/mismatch-doc|{modes[tag]}|B={args.budget}"
        result = load_result(args.runs_dir, tag)
        if result is None or clean not in result.get("metrics", {}):
            print(f"{tag:<10}{labels.get(tag, ''):<20}{'(no result yet)':>44}")
            continue
        available.append(tag)
        metrics = result["metrics"]
        em = 100 * metrics[clean]["em"]
        sub = 100 * metrics[clean]["substring"]
        has_floor = floor in metrics
        low = 100 * metrics[floor]["em"] if has_floor else float("nan")
        curve = validation_curve(args.runs_dir, tag)
        best = max((em for _, em in curve), default=None)
        print(f"{tag:<10}{labels.get(tag, ''):<20}{em:>8.2f}{sub:>8.2f}"
              f"{low:>8.2f}{em - low:>10.2f}"
              f"{100 * best if best is not None else float('nan'):>10.2f}")

    if args.baseline not in available:
        print("\nbaseline has no result yet; nothing to test against")
        return

    print(f"\npaired McNemar against {args.baseline}"
          f" ({labels.get(args.baseline, 'baseline')})")
    print(f"{'run':<10}{'metric':<11}{'delta':>8}{'win':>6}{'loss':>6}{'p':>12}")
    print("-" * 55)
    for tag in available:
        if tag == args.baseline:
            continue
        directions = {}
        for metric in ("em", "substring"):
            a = load_predictions(args.runs_dir, tag, args.split, modes[tag], args.budget)
            b = load_predictions(args.runs_dir, args.baseline, args.split,
                                 baseline_mode, args.budget)
            if not a or not b:
                continue
            ids = [i for i in a if i in b]
            deltas = [a[i][metric] - b[i][metric] for i in ids]
            wins, losses, p = mcnemar(deltas)
            delta = 100 * sum(deltas) / len(deltas)
            directions[metric] = delta
            print(f"{tag:<10}{metric:<11}{delta:>+8.2f}{wins:>6}{losses:>6}"
                  f"{p:>12.2e}  {stars(p)}")
        if len(directions) == 2 and (directions["em"] > 0) != (directions["substring"] > 0):
            print(f"{'':<10}EM and substring disagree in direction -- report both, "
                  "not one")
        print()


if __name__ == "__main__":
    main()
