"""Gather every arm run into one auditable table, with paired tests.

Each run writes its own ``result.json`` under ``/data02/quro/runs``, which is
enough to read a single number and not enough to compare any two. This collects
them into ``results/arm_matrix.json`` -- the arm each run *actually implements*
(``arm_label``, not the tag), the training setting that makes a column
comparable, the mismatch-document floor that every accuracy has to be read
against, and paired McNemar tests over the per-item predictions.

The floor is not decoration. On TriviaQA a 7B answers 59% of the questions with
the wrong documents in front of it, so a 68% score represents 9 points of
evidence use, not 68. Reporting accuracy without its floor is the single easiest
way to overstate this method.

    python scripts/collect_arm_matrix.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from math import comb

RUNS_DEFAULT = "/data02/quro/runs"

# tag -> (dataset, eval split name, training setting label).  The setting matters:
# qdrop=1.0 never shows the decoder the question during training, so its D0 is
# out of distribution and must not be put in the same column as a qdrop=0 run.
REGISTRY = [
    # TriviaQA, trained with the question always removed (qdrop=1.0)
    ("w1_A0", "triviaqa", "trivia", "qdrop1.0"),
    ("w1_A1", "triviaqa", "trivia", "qdrop1.0"),
    ("w1_C0", "triviaqa", "trivia", "qdrop1.0"),
    ("w1_C1", "triviaqa", "trivia", "qdrop1.0"),
    # HotpotQA, same setting, so the two datasets are directly comparable
    ("hp1_A0", "hotpotqa", "dev", "qdrop1.0"),
    ("hp2_A1", "hotpotqa", "dev", "qdrop1.0"),
    ("hp1_C0", "hotpotqa", "dev", "qdrop1.0"),
    ("hp1_C1", "hotpotqa", "dev", "qdrop1.0"),
    ("hp1_S", "hotpotqa", "dev", "qdrop1.0"),
    # HotpotQA, standard training (question in the prompt) -- the main table.
    # Trained with budget buckets 4,8 and dropout on.
    ("hp2d0_C1", "hotpotqa", "dev", "qdrop0"),
    ("hp2d0_S", "hotpotqa", "dev", "qdrop0"),
    ("hp2d0_P", "hotpotqa", "dev", "qdrop0"),
    # B sweep: single budget, dropout off, so the points are matched to each
    # other.  The hp2d0 runs above are NOT matched anchors for this curve -- they
    # share a budget across buckets, and slot self-attention has no budget mask,
    # so B=8 there is not the same system as bs8 here (W5).
    ("bs8_C1", "hotpotqa", "dev", "qdrop0/fixedB"),
    ("bs16_C1", "hotpotqa", "dev", "qdrop0/fixedB"),
    ("bs32_C1", "hotpotqa", "dev", "qdrop0/fixedB"),
    ("bs8S_S", "hotpotqa", "dev", "qdrop0/fixedB"),
    ("bs16S_S", "hotpotqa", "dev", "qdrop0/fixedB"),
    ("bs32S_S", "hotpotqa", "dev", "qdrop0/fixedB"),
    ("bs32_C0", "hotpotqa", "dev", "qdrop0/fixedB"),
    # Historical TriviaQA runs.  These predate the arm taxonomy, and their "A" is
    # A1: the cosine prior was on.  They also used the mismatch-*query* control,
    # so they carry no evidence floor and their evidence use cannot be recovered.
    ("gonogo_C_D0", "triviaqa", "trivia", "qdrop0"),
    ("gonogo_A_D0", "triviaqa", "trivia", "qdrop0"),
    ("gonogo_S_D0", "triviaqa", "trivia", "qdrop0"),
    ("gonogo_P_D0", "triviaqa", "trivia", "qdrop0"),
    ("d1_C_full", "triviaqa", "trivia", "qdrop1.0"),
    ("d1_A_full", "triviaqa", "trivia", "qdrop1.0"),
]

# Comparisons worth a paired test, as (dataset, setting, mode, arm_a, arm_b).
PAIRED = [
    ("hotpotqa", "qdrop0", "D0", "hp2d0_C1", "hp2d0_S"),
    ("hotpotqa", "qdrop0", "D0", "hp2d0_P", "hp2d0_C1"),
    ("hotpotqa", "qdrop0", "D0", "hp2d0_P", "hp2d0_S"),
    ("hotpotqa", "qdrop1.0", "D0", "hp1_C1", "hp1_S"),
    ("hotpotqa", "qdrop1.0", "D1", "hp1_C1", "hp1_S"),
    ("hotpotqa", "qdrop1.0", "D1", "hp1_C1", "hp1_C0"),
    ("hotpotqa", "qdrop1.0", "D1", "hp1_C0", "hp1_A0"),
    ("hotpotqa", "qdrop1.0", "D1", "hp2_A1", "hp1_A0"),
    ("triviaqa", "qdrop1.0", "D1", "w1_C1", "w1_C0"),
    ("triviaqa", "qdrop1.0", "D1", "w1_C0", "w1_A0"),
    ("triviaqa", "qdrop1.0", "D1", "w1_A1", "w1_A0"),
    ("triviaqa", "qdrop1.0", "D1", "w1_C1", "w1_A1"),
    # B sweep: is the learnable readout's margin over the cosine rule a
    # low-budget phenomenon?
    ("hotpotqa", "qdrop0/fixedB", "D0", "bs8_C1", "bs8S_S"),
    ("hotpotqa", "qdrop0/fixedB", "D0", "bs16_C1", "bs16S_S"),
    ("hotpotqa", "qdrop0/fixedB", "D0", "bs32_C1", "bs32S_S"),
    ("hotpotqa", "qdrop0/fixedB", "D0", "bs32_C1", "bs32_C0"),
]

# Runs whose budget is not 8; used to line up predictions files and to label the
# curve.  Anything absent is B=8.
BUDGETS = {"bs16_C1": 16, "bs32_C1": 32, "bs16S_S": 16, "bs32S_S": 32, "bs32_C0": 32}


def mcnemar(deltas):
    """Two-sided exact test over the discordant pairs."""
    wins = sum(1 for d in deltas if d > 0)
    losses = sum(1 for d in deltas if d < 0)
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    tail = sum(comb(n, i) for i in range(min(wins, losses) + 1))
    return wins, losses, min(1.0, 2 * tail / 2 ** n)


def load_predictions(runs, tag, split, mode):
    budget = BUDGETS.get(tag, 8)
    name = f"predictions_{split.replace('/', '_')}_{mode}_B{budget}.json"
    path = os.path.join(runs, tag, name)
    if not os.path.exists(path):
        return None
    return {row["id"]: row for row in json.load(open(path, encoding="utf-8"))}


def collect(runs):
    out = {}
    for tag, dataset, split, setting in REGISTRY:
        path = os.path.join(runs, tag, "result.json")
        if not os.path.exists(path):
            print(f"  skip {tag}: no result.json")
            continue
        record = json.load(open(path, encoding="utf-8"))
        entry = {
            "dataset": dataset, "split": split, "setting": setting,
            # arm_label() re-derives what the config implements; the historical
            # runs predate it and are labelled from their raw fields below.
            "arm": record.get("arm"),
            "output_query_mode": record.get("output_query_mode"),
            "cosine_prior": record.get("cosine_prior"),
            "readout": record.get("readout"),
            "query_text_dropout": record.get("query_text_dropout"),
            "modes": {},
        }
        if entry["arm"] is None:
            # Pre-taxonomy run.  Recover what can be recovered from config.json
            # rather than inventing a label.
            cfg_path = os.path.join(runs, tag, "config.json")
            if os.path.exists(cfg_path):
                r = json.load(open(cfg_path, encoding="utf-8"))["readout"]
                entry["output_query_mode"] = r.get("output_query_mode")
                entry["cosine_prior"] = r.get("cosine_prior")
                entry["readout"] = r.get("kind")
                entry["arm_note"] = (
                    "predates the arm taxonomy; cosine_prior was not recorded in "
                    "config at this commit" if r.get("cosine_prior") is None else
                    "labelled retrospectively from config.json")

        for key, agg in record.get("metrics", {}).items():
            name, mode, budget = key.split("|")
            b = int(budget.split("=")[1])
            # The B sweep evaluates at one budget per run; the earlier runs
            # evaluate at several.  Key by budget so both fit the same shape.
            entry.setdefault("budget", b)
            control = "mismatch-doc" if "mismatch-doc" in name else (
                "mismatch-q" if "mismatch-q" in name else "clean")
            entry["modes"].setdefault(f"{mode}|B={b}", {})[control] = {
                "em": agg["em"], "substring": agg["substring"], "f1": agg["f1"],
                "n": agg["n"],
                "constant_baseline_em": agg.get("constant_baseline_em"),
                "mean_decoder_input_tokens": agg.get("mean_decoder_input_tokens"),
                "xi_eff": agg.get("xi_eff"),
            }
        # Evidence use: accuracy above the wrong-document control.  Only defined
        # where that control was actually run -- the gonogo runs used a mismatch
        # *query* instead, which bounds something else entirely.
        for mode, controls in entry["modes"].items():
            if "clean" in controls and "mismatch-doc" in controls:
                controls["evidence_em"] = round(
                    controls["clean"]["em"] - controls["mismatch-doc"]["em"], 6)
        out[tag] = entry
    return out


def run_paired(runs, table):
    results = []
    for dataset, setting, mode, tag_a, tag_b in PAIRED:
        if tag_a not in table or tag_b not in table:
            continue
        split = table[tag_a]["split"]
        A = load_predictions(runs, tag_a, split, mode)
        B = load_predictions(runs, tag_b, split, mode)
        if not A or not B:
            continue
        ids = [i for i in A if i in B]
        row = {"dataset": dataset, "setting": setting, "mode": mode,
               "a": tag_a, "b": tag_b, "arm_a": table[tag_a]["arm"],
               "arm_b": table[tag_b]["arm"], "n": len(ids)}
        for metric in ("em", "substring"):
            deltas = [A[i][metric] - B[i][metric] for i in ids]
            wins, losses, p = mcnemar(deltas)
            row[metric] = {"delta": round(100 * sum(deltas) / len(deltas), 4),
                           "wins": wins, "losses": losses, "p": p}
        results.append(row)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS_DEFAULT)
    ap.add_argument("--out", default="results/arm_matrix.json")
    args = ap.parse_args()

    table = collect(args.runs)
    paired = run_paired(args.runs, table)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": commit,
        "note": ("EM/substring are fractions. evidence_em is clean minus the "
                 "mismatch-document control and is the only number that reflects "
                 "evidence use; runs whose control was mismatch-query have none. "
                 "qdrop1.0 and qdrop0 columns are not comparable: qdrop1.0 never "
                 "shows the decoder the question during training, so its D0 is "
                 "out of distribution."),
        "runs": table,
        "paired_tests": paired,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"wrote {args.out}: {len(table)} runs, {len(paired)} paired tests")


if __name__ == "__main__":
    main()
