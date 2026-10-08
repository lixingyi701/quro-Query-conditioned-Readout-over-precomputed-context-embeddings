"""Paired SQX/SQ and 2x2 query/cross-document validation; never intersects IDs.

--run SEED:ARM=DIR accepts S0, SQ, S0X, SQX. Two-arm historical analysis is
available with --legacy; its missing training controls are explicitly reported.
Scores and confidence intervals in JSON are fractions, printed values are pp.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


ARMS = {"S0": ("none", False), "SQ": ("conditioned", False),
        "S0X": ("none", True), "SQX": ("conditioned", True)}
METRICS = ("f1", "em", "substring")
CONTRASTS = {
    "SQX-SQ": {"SQX": 1, "SQ": -1},
    "S0X-S0": {"S0X": 1, "S0": -1},
    "SQ-S0": {"SQ": 1, "S0": -1},
    "SQX-S0X": {"SQX": 1, "S0X": -1},
    "query_crossdoc_interaction": {"SQX": 1, "SQ": -1, "S0X": -1, "S0": 1},
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predictions(path):
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty predictions: {path}")
    items = {}
    for row in rows:
        key = str(row["id"])
        if key in items:
            raise ValueError(f"duplicate prediction ID: {key}")
        for metric in METRICS:
            value = float(row[metric])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid {metric}: {key}")
        items[key] = row
    return items


def paired_summary(values, resamples=2000, seed=0):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or resamples < 100:
        raise ValueError("finite paired scores and at least 100 resamples required")
    rng = np.random.default_rng(seed)
    samples = []
    for offset in range(0, resamples, 128):
        indices = rng.integers(0, len(x), size=(min(128, resamples-offset), len(x)))
        samples.extend(x[indices].mean(axis=1).tolist())
    return {"delta": float(x.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "n": len(x), "positive": int((x > 1e-12).sum()), "negative": int((x < -1e-12).sum())}


def check_metadata(records, training_seed, legacy):
    common = ("query_encoder_kind", "query_representation", "projector_hidden",
              "projector_conditioning", "projector_attention_dim", "projector_heads",
              "max_query_len", "generator_lora_init", "train_decoder_input_mode",
              "offline_m", "offline_compressor", "offline_compr_rate", "query_text_dropout")
    reference = next(iter(records.values()))
    historical_defaults = {"projector_fusion": "none", "support_head": False, "support_loss_weight": 0}
    for arm, record in records.items():
        mode, cross = ARMS[arm]
        expected = {"readout": "shared_projector", "projector_query_mode": mode,
                    "projector_cross_document": cross, "projector_fusion": "none",
                    "support_head": False, "support_loss_weight": 0,
                    "train_decoder_input_mode": "D0", "generator_lora_init": "frozen"}
        for key, value in expected.items():
            actual = record.get(key, historical_defaults.get(key) if legacy else None)
            if actual != value:
                raise ValueError(f"wrong {arm} metadata: {key}")
        for key in common:
            if key not in record or record[key] != reference.get(key):
                raise ValueError(f"missing or unmatched metadata: {key}")
        if not legacy and (not record.get("evaluation_protocol")
                           or record["evaluation_protocol"] != reference.get("evaluation_protocol")):
            raise ValueError("missing or mismatched evaluation protocol")
    provenances = [r.get("data_order_provenance") for r in records.values()]
    controlled = all(p is not None for p in provenances)
    if controlled:
        for record, p in zip(records.values(), provenances):
            protocol = p.get("protocol") or {}
            fields = ("seed", "data_order_seed", "train_sha256", "cache_manifest_sha256",
                      "protocol", "data_limits", "generator_path", "fresh_start", "completed_steps",
                      "microbatches", "examples", "order_sha256")
            if any(p.get(k) is None or p[k] != provenances[0].get(k) for k in fields):
                controlled = False
            evaluation = record.get("evaluation_protocol") or {}
            if not legacy and (evaluation.get("cache_manifest_sha256") != p.get("cache_manifest_sha256")
                    or evaluation.get("generator_path") != p.get("generator_path")
                    or evaluation.get("gen_max_new_tokens") != protocol.get("gen_max_new_tokens")
                    or any(evaluation.get(k) != (p.get("data_limits") or {}).get(k)
                           for k in ("max_docs", "max_query_len"))):
                controlled = False
            if (p.get("seed") != training_seed or p.get("fresh_start") is not True
                    or not protocol.get("steps") or p.get("completed_steps") != protocol["steps"]
                    or not p.get("microbatches") or not p.get("examples")
                    or p.get("microbatches") != protocol["steps"] * protocol.get("grad_accum", 0)
                    or p.get("examples") != p.get("microbatches") * protocol.get("batch_size", 0)):
                controlled = False
    if not controlled and not legacy:
        raise ValueError("training order/protocol missing or mismatched; rerun matched arms or explicitly use --legacy")
    return controlled


def analyze(runs, data_file, split="dev", resamples=2000, bootstrap_seed=0, legacy=False):
    meta = {}
    for line in Path(data_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row["id"])
        if key in meta:
            raise ValueError(f"duplicate dataset ID: {key}")
        meta[key] = row
    if not runs or any(len(arms) < 2 for arms in runs.values()):
        raise ValueError("supply at least two arms per training seed")
    arm_set = set(next(iter(runs.values())))
    if any(set(arms) != arm_set for arms in runs.values()):
        raise ValueError("all seeds must contain the same arms")
    valid_contrasts = {k: c for k, c in CONTRASTS.items() if set(c) <= arm_set}
    if not valid_contrasts:
        raise ValueError("no supported contrast in supplied arms")
    normal, swapped, audit, results = {}, {}, {}, {}
    ids = None
    control_presence = []
    reference = None
    for seed, arms in sorted(runs.items()):
        normal[seed], swapped[seed] = {}, {}
        records = {arm: read_json(Path(root)/"result.json") for arm, root in arms.items()}
        audit[str(seed)] = {"matched_training": check_metadata(records, seed, legacy),
                            "missing_historical_fields": {a: [k for k in (
                                "projector_fusion", "support_head", "support_loss_weight") if k not in r]
                                for a, r in records.items()}}
        results[seed] = records
        for arm, root in arms.items():
            items = predictions(Path(root)/f"predictions_{split}_D0_Bfull.json")
            if ids is None:
                ids, reference = sorted(items), items
            if set(items) != set(ids):
                raise ValueError("prediction ID sets differ; refusing silent intersection")
            for key in ids:
                if key not in meta or meta[key].get("hop_type") not in {"bridge", "comparison"}:
                    raise ValueError(f"missing dataset row or hop_type: {key}")
                expected_golds = meta[key].get("answers") or [meta[key]["answer"]]
                if (items[key].get("golds") != expected_golds
                        or items[key].get("query") != meta[key]["query"].strip()):
                    raise ValueError(f"dataset/prediction mismatch: {key}")
                for field in ("query", "golds", "decoder_query", "readout_query"):
                    if items[key].get(field) != reference[key].get(field):
                        raise ValueError(f"different normal {field}: {key}")
                if items[key].get("decoder_query") or items[key].get("readout_query"):
                    raise ValueError("normal predictions unexpectedly contain a query intervention")
            normal[seed][arm] = items
            path = Path(root)/f"predictions_{split}_mismatch-q_D0_Bfull.json"
            control_presence.append(path.exists())
            if path.exists():
                swapped[seed][arm] = predictions(path)
                if set(swapped[seed][arm]) != set(ids):
                    raise ValueError("query-control ID sets differ")
                for key in ids:
                    for field in ("query", "golds", "decoder_query"):
                        if swapped[seed][arm][key].get(field) != items[key].get(field):
                            raise ValueError(f"query control changed decoder question or labels: {key}")
    if any(control_presence) and not all(control_presence):
        raise ValueError("query controls must be present for every supplied run or none")
    if not legacy:
        seed_reference = next(iter(results.values()))
        for records in results.values():
            for arm, record in records.items():
                left, right = record["data_order_provenance"], seed_reference[arm]["data_order_provenance"]
                for key in ("train_sha256", "cache_manifest_sha256", "protocol", "data_limits", "generator_path"):
                    if left[key] != right[key]:
                        raise ValueError(f"different training protocol across seeds: {key}")
                if record["evaluation_protocol"] != seed_reference[arm]["evaluation_protocol"]:
                    raise ValueError("different evaluation protocol across seeds")
    # Train seeds are reported separately; averaging their per-question scores
    # does not increase the number of independently resampled questions.
    average = {arm: {key: {metric: float(np.mean([normal[s][arm][key][metric] for s in runs]))
                           for metric in METRICS} for key in ids} for arm in arm_set}
    report = {"split": split, "units": "fraction", "training_seeds": sorted(runs), "audit": audit,
              "legacy_exploratory": legacy, "bootstrap_resamples": resamples,
              "uncertainty": "paired questions only; CIs exclude training-seed uncertainty",
              "parameters": {str(s): {a: r.get("parameters") for a, r in rs.items()}
                             for s, rs in results.items()}, "groups": {}}
    subsets = {"all": ids, **{hop: [key for key in ids if meta[key]["hop_type"] == hop]
                              for hop in ("bridge", "comparison")}}
    for group, keys in subsets.items():
        if not keys:
            continue
        output = {"n": len(keys), "by_seed": {}, "seed_mean": {}}
        for label, scores in [(str(s), normal[s]) for s in sorted(runs)] + [("seed_mean", average)]:
            entry = {"qa": {arm: {m: float(np.mean([scores[arm][i][m] for i in keys]))
                                   for m in METRICS} for arm in sorted(arm_set)}, "contrasts": {}}
            for name, coeffs in valid_contrasts.items():
                entry["contrasts"][name] = {m: paired_summary(
                    [sum(weight*scores[a][i][m] for a, weight in coeffs.items()) for i in keys],
                    resamples, bootstrap_seed) for m in METRICS}
            if label == "seed_mean":
                output["seed_mean"] = entry
            else:
                output["by_seed"][label] = entry
                if all(control_presence):
                    entry["query_swap_drop"] = {arm: paired_summary(
                        [normal[int(label)][arm][i]["f1"]-swapped[int(label)][arm][i]["f1"] for i in keys],
                        resamples, bootstrap_seed) for arm in sorted(arm_set)}
        report["groups"][group] = output
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="SEED:ARM=DIR")
    parser.add_argument("--data_file", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=0)
    parser.add_argument("--legacy", action="store_true", help="historical exploratory comparison; relax training-provenance checks")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    runs = {}
    for item in args.run:
        key, root = item.split("=", 1)
        seed_text, arm = key.split(":", 1)
        seed = int(seed_text)
        if arm not in ARMS or arm in runs.setdefault(seed, {}):
            raise ValueError("unknown or duplicated seed/arm")
        runs[seed][arm] = root
    destination = Path(args.output).resolve()
    inputs = {Path(args.data_file).resolve()}
    inputs.update(p.resolve() for arms in runs.values() for root in arms.values() for p in Path(root).glob("*.json"))
    if destination in inputs or destination.exists():
        raise ValueError("output must be a new file outside the input run JSON files")
    report = analyze(runs, args.data_file, args.split, args.resamples, args.bootstrap_seed, args.legacy)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Historical exploratory comparison" if args.legacy else "Matched training comparison")
    for group, result in report["groups"].items():
        print(f"{group}: n={result['n']}")
        for seed, entry in list(result["by_seed"].items()) + [("seed_mean", result["seed_mean"])]:
            for name, values in entry["contrasts"].items():
                value = values["f1"]
                lo, hi = value["ci95"]
                print(f"  {seed:>9} {name:<26} F1 {100*value['delta']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] pp")
    print("CI resampling unit: question; no training-seed uncertainty estimate.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError) as error:
        sys.exit(str(error))
