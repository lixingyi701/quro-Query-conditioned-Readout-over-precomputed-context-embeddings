"""Paired question bootstrap for the four input interventions in one eval run."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MODES = ("original", "scale_only", "direction_only", "full")
CONTRASTS = {
    "full_minus_original": (-1, 0, 0, 1),
    "scale_minus_original": (-1, 1, 0, 0),
    "direction_minus_original": (-1, 0, 1, 0),
    "full_minus_scale": (0, -1, 0, 1),
    "full_minus_direction": (0, 0, -1, 1),
    "interaction": (1, -1, -1, 1),
}
METRICS = ("em", "f1", "substring", "content_logprob_mean", "eos_after_gold_logprob",
           "generated_content_tokens", "eos_reached", "hit_generation_cap")


def load_run(root):
    root = Path(root)
    with open(root/"result.json", encoding="utf-8") as handle:
        meta = json.load(handle)
    with open(root/"predictions.jsonl", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not meta.get("completed") or tuple(meta.get("modes", [])) != MODES:
        raise ValueError("need a completed four-arm input attribution run")
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("prediction IDs must be nonempty and unique")
    digest = hashlib.sha256(json.dumps([row["id"] for row in rows]).encode()).hexdigest()
    if digest != meta["evaluated_ids_sha256"]:
        raise ValueError("prediction ID order/content differs from the recorded run")
    for row in rows:
        if set(row["modes"]) != set(MODES):
            raise ValueError(f"missing intervention for {row['id']}")
        lengths = {row["modes"][mode]["content_tokens"] for mode in MODES}
        if len(lengths) != 1:
            raise ValueError("gold content lengths differ across interventions")
        for mode in MODES:
            for metric in METRICS:
                value = float(row["modes"][mode][metric])
                if not np.isfinite(value) or (metric in {"em", "f1", "substring", "eos_reached", "hit_generation_cap"}
                                              and not 0 <= value <= 1):
                    raise ValueError(f"invalid {metric} for {row['id']}/{mode}")
    for mode in MODES:
        if meta["qa"][mode]["n"] != len(rows):
            raise ValueError("result and prediction counts disagree")
        for metric in ("em", "f1", "substring"):
            actual = sum(row["modes"][mode][metric] for row in rows)/len(rows)
            if not np.isclose(actual, meta["qa"][mode][metric], rtol=0, atol=1e-10):
                raise ValueError("result and prediction QA metrics disagree")
    return meta, rows


def paired_summary(values, bootstrap=2000, seed=0):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or bootstrap < 2:
        raise ValueError("paired bootstrap needs finite values and at least two resamples")
    rng = np.random.default_rng(seed)
    resamples = []
    # Bound index memory for full test sets.
    chunk = max(1, min(128, 1000000//len(x)))
    for offset in range(0, bootstrap, chunk):
        indices = rng.integers(0, len(x), size=(min(chunk, bootstrap-offset), len(x)))
        resamples.extend(x[indices].mean(axis=1).tolist())
    return {"difference": float(x.mean()), "ci95": np.quantile(resamples, [.025, .975]).tolist(), "n": len(x)}


def analyze_rows(rows, bootstrap=2000, seed=0):
    names = list(METRICS)
    margin_presence = ["content_candidate_margin" in row["modes"][mode] for row in rows for mode in MODES]
    if any(margin_presence) and not all(margin_presence):
        raise ValueError("candidate margin missing in some paired arms")
    if all(margin_presence):
        names.append("content_candidate_margin")
    output = {}
    for contrast, weights in CONTRASTS.items():
        output[contrast] = {}
        for metric in names:
            differences = [sum(w*float(row["modes"][mode][metric]) for mode, w in zip(MODES, weights)) for row in rows]
            output[contrast][metric] = paired_summary(differences, bootstrap, seed)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    meta, rows = load_run(args.run)
    report = {"units": "QA fractions; log-probability in nats; length in tokens",
              "bootstrap": args.bootstrap, "bootstrap_seed": args.seed,
              "uncertainty": "paired questions only; excludes training-seed uncertainty",
              "checkpoint_sha256": meta["checkpoint_sha256"],
              "toy_smoke_only": meta.get("toy_smoke_only", False),
              "overall": analyze_rows(rows, args.bootstrap, args.seed), "exploratory_strata": {}}
    groups = {}
    for row in rows:
        for field, value in row.get("strata", {}).items():
            groups.setdefault(f"{field}={value}", []).append(row)
    for name, group in groups.items():
        report["exploratory_strata"][name] = analyze_rows(group, args.bootstrap, args.seed)
    out = Path(args.out) if args.out else Path(args.run)/"paired_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    for name, results in report["overall"].items():
        f1 = results["f1"]
        print(f"{name}: {100*f1['difference']:+.3f} F1 pp, "
              f"95% CI [{100*f1['ci95'][0]:+.3f}, {100*f1['ci95'][1]:+.3f}], n={f1['n']}")
    return report


if __name__ == "__main__":
    main()
