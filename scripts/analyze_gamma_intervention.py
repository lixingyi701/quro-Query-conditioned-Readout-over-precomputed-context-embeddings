"""Frozen-gamma control and gamma interventions: paired QA differences.

Standard library only; reuses the fusion analyzer's strict pairing and bootstrap.
Two parts:
  arms          QA of trained arms against the frozen-gamma (G0) control and HeadE.
  interventions per checkpoint, normal minus gamma-zero / gamma-swap / mismatch-q,
                plus the paired gamma and E changes recorded in result.json.
JSON scores/differences are fractions; the printed tables use percentage points.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_projector_fusion import METRICS, check_pair, paired_summary, read_json, read_predictions

INTERVENTIONS = ("gamma-zero", "gamma-swap", "mismatch-q")
CHANGE_KEYS = ("change_gamma_rel", "change_gamma_cosine", "change_e_over_delta_ref",
               "change_e_over_memory")


def predictions(run, split, variant=None):
    name = split if variant is None else f"{split}_{variant}"
    return read_predictions(os.path.join(run, f"predictions_{name}_D0_Bfull.json"))


def subsets(rows, meta):
    if not meta:
        return {}
    return {h: [k for k in rows if meta[k].get("hop_type") == h] for h in ("bridge", "comparison")}


def diff(a, b, keys, metric, bootstrap, seed):
    return paired_summary([float(a[k][metric]) - float(b[k][metric]) for k in sorted(keys)],
                          bootstrap, seed)


def compare(a, b, meta, bootstrap, seed, query_control=False):
    check_pair(a, b, query_control=query_control)
    out = {m: diff(a, b, a, m, bootstrap, seed) for m in METRICS}
    for name, keys in subsets(a, meta).items():
        out[f"f1_{name}"] = diff(a, b, keys, "f1", bootstrap, seed)
    return out


def fmt(s):
    lo, hi = s["ci95"]
    return f"{100*s['difference']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", default=[], metavar="NAME=RUN",
                        help="trained arm run directory; repeatable")
    parser.add_argument("--control", required=True, metavar="NAME=RUN",
                        help="frozen-gamma control run directory")
    parser.add_argument("--reference", action="append", default=[], metavar="NAME=RUN",
                        help="extra baselines (e.g. HeadE), compared against the control")
    parser.add_argument("--intervention", action="append", default=[], metavar="NAME=RUN",
                        help="eval-only run with --query_control --gamma_control; repeatable")
    parser.add_argument("--dev", help="original dev jsonl with hop_type, for subsets")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json")
    args = parser.parse_args()

    def named(value):
        name, _, run = value.partition("=")
        if not run:
            raise SystemExit(f"expected NAME=RUN, got {value}")
        return name, run

    meta = {}
    if args.dev:
        with open(args.dev, encoding="utf-8") as handle:
            meta = {r["id"]: r for r in map(json.loads, handle)}
    report = {"arms": {}, "interventions": {}}
    control_name, control_run = named(args.control)
    control = predictions(control_run, args.split)
    print(f"QA against {control_name} (row minus control), pp")
    for value in args.arm + args.reference:
        name, run = named(value)
        result = compare(predictions(run, args.split), control, meta, args.bootstrap, args.seed)
        report["arms"][f"{name}-{control_name}"] = result
        print(f"  {name:>10}: " + "  ".join(f"{k} {fmt(v)}" for k, v in result.items()))

    print("\nInterventions (normal minus intervened), pp; paired changes from result.json")
    for value in args.intervention:
        name, run = named(value)
        normal = predictions(run, args.split)
        metrics = read_json(os.path.join(run, "result.json"))["metrics"]
        entry = {}
        for variant in INTERVENTIONS:
            other = predictions(run, args.split, variant)
            result = compare(normal, other, meta, args.bootstrap, args.seed,
                             query_control=variant == "mismatch-q")
            changes = metrics[f"{args.split}/{variant}|D0|B=full"]
            result["changes"] = {k: changes.get(k) for k in CHANGE_KEYS}
            entry[variant] = result
            shown = "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                              for k, v in result["changes"].items())
            print(f"  {name:>10} {variant:>10}: f1 {fmt(result['f1'])}  em {fmt(result['em'])}"
                  + "".join(f"  {k} {fmt(result[k])}" for k in result if k.startswith("f1_"))
                  + f"\n{'':>24}{shown}")
        report["interventions"][name] = entry
    if args.output_json:
        if os.path.exists(args.output_json):
            raise SystemExit(f"refusing to overwrite {args.output_json}")
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
