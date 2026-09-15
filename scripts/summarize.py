"""Collect run results into one table and apply the go/no-go criteria.

The thresholds live here, in code, fixed before the runs finish.  The point of
writing them down is that "is this good enough?" stops being a judgement call
made after seeing the numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import paths

# Fixed in advance; see article/QURO_V0.1_IMPLEMENTATION_PLAN.md §6.
MIN_C_MINUS_A_EM = 0.03          # query conditioning must be worth >= 3 EM points
MIN_MISMATCH_DROP_EM = 0.05      # C must lose >= 5 EM points on a wrong query
MAX_C_BELOW_P_EM = 0.03          # C at B=8 may trail PISCO at B=K*m by <= 3 points


def load_runs(root: str) -> dict:
    out = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "result.json"))):
        with open(path, encoding="utf-8") as f:
            out[os.path.basename(os.path.dirname(path))] = json.load(f)
    return out


def get(runs, tag, key, metric="em"):
    run = runs.get(tag)
    if not run:
        return None
    value = run["metrics"].get(key)
    return None if value is None else value.get(metric)


def print_table(runs):
    keys = sorted({k for run in runs.values() for k in run["metrics"]})
    width = max([len(t) for t in runs] + [12])
    print(f"\n{'run':<{width}}  {'split|mode|B':<26} {'EM':>7} {'F1':>7} {'sub':>7} "
          f"{'B_tok':>7} {'xi_eff':>8}")
    print("-" * (width + 68))
    for tag in sorted(runs):
        for key in keys:
            row = runs[tag]["metrics"].get(key)
            if not row:
                continue
            xi = f"{row['xi_eff']:.1f}" if row.get("xi_eff") else "-"
            per_query = row["readout_tokens"] / max(1, row["n"])
            print(f"{tag:<{width}}  {key:<26} {row['em']:>6.2%} {row['f1']:>7.3f} "
                  f"{row['substring']:>6.2%} {per_query:>7.1f} {xi:>8}")


def verdict(runs, mode="D0", budget=8, split="dev"):
    key = f"{split}|{mode}|B={budget}"
    mismatch = f"{split}/mismatch-q|{mode}|B={budget}"
    c = get(runs, f"gonogo_C_{mode}", key) or get(runs, "gonogo_C_qdrop", key)
    a = get(runs, f"gonogo_A_{mode}", key) or get(runs, "gonogo_A_qdrop", key)
    s = get(runs, f"gonogo_S_{mode}", key) or get(runs, "gonogo_S_qdrop", key)
    p = get(runs, f"gonogo_P_{mode}", key)
    c_mis = get(runs, f"gonogo_C_{mode}", mismatch) or get(runs, "gonogo_C_qdrop", mismatch)

    print(f"\n=== go/no-go verdict ({split}, {mode}, B={budget}) ===")
    checks = []
    if c is None or a is None:
        print("  arms C and A are both required; run stage 1 first")
        return False
    checks.append((f"C - A = {c - a:+.2%} (need >= {MIN_C_MINUS_A_EM:.0%})",
                   c - a >= MIN_C_MINUS_A_EM))
    if s is not None:
        checks.append((f"C - S = {c - s:+.2%} (need > 0)", c > s))
    else:
        print("  [skip] similarity_topb arm missing")
    if c_mis is not None:
        checks.append((f"mismatch-query drop = {c - c_mis:+.2%} "
                       f"(need >= {MIN_MISMATCH_DROP_EM:.0%})",
                       c - c_mis >= MIN_MISMATCH_DROP_EM))
    else:
        print("  [skip] no mismatch-query run (pass --query_control)")
    if p is not None:
        checks.append((f"P - C = {p - c:+.2%} at unmatched budget "
                       f"(need <= {MAX_C_BELOW_P_EM:.0%})", p - c <= MAX_C_BELOW_P_EM))
    else:
        print("  [skip] pisco_direct reference arm missing")

    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    decision = all(ok for _, ok in checks)
    print(f"\n  ==> {'GO' if decision else 'NO-GO'}: "
          + ("proceed to the main experiments" if decision
             else "see QURO_EXPERIMENTAL_DESIGN.md §9 before spending more compute"))
    return decision


def query_text_analysis(runs, budget=8, split="dev"):
    """How much does removing the plain-text question cost each arm?"""
    print(f"\n=== does the decoder still need the question in plain text? (B={budget}) ===")
    any_row = False
    for tag in sorted(runs):
        d0 = get(runs, tag, f"{split}|D0|B={budget}")
        d1 = get(runs, tag, f"{split}|D1|B={budget}")
        if d0 is None or d1 is None:
            continue
        any_row = True
        print(f"  {tag:<20} D0={d0:.2%}  D1={d1:.2%}  delta={d1 - d0:+.2%}")
    if not any_row:
        print("  no run evaluated both D0 and D1 (use --eval_input_modes D0,D1)")
        return
    print("  A small delta for C means the readout absorbed the query; a large one\n"
          "  means the soft tokens are an evidence bag and the decoder does the matching.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=paths.RUNS_DIR)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--mode", default="D0")
    ap.add_argument("--split", default="dev")
    args = ap.parse_args()

    runs = load_runs(args.runs)
    if not runs:
        print(f"no result.json under {args.runs}")
        return
    print_table(runs)
    verdict(runs, args.mode, args.budget, args.split)
    query_text_analysis(runs, args.budget, args.split)


if __name__ == "__main__":
    main()
