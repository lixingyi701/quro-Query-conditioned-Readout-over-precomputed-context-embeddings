"""Paired FiLM minus additive QA analysis, with query swaps and RMS diagnostics.

Standard library only. Refuses silent ID intersections or different prompts.
JSON scores/differences are fractions; the printed table uses percentage points.
"""
import argparse
import json
import math
from pathlib import Path
import random


METRICS = ("em", "f1", "substring")


def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_predictions(path):
    rows = read_json(path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"predictions must be a nonempty array: {path}")
    result = {}
    for row in rows:
        key = row["id"]
        if key in result:
            raise ValueError(f"duplicate prediction id: {key}")
        for name in METRICS:
            value = float(row[name])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid {name} for {key}")
        result[key] = row
    return result


def check_pair(a, b, query_control=False):
    if set(a) != set(b):
        raise ValueError("prediction ID sets differ; refusing an intersection-only comparison")
    for key in a:
        for field in ("query", "golds", "decoder_query"):
            if a[key].get(field) != b[key].get(field):
                raise ValueError(f"different {field} for {key}")
        for field in ("doc_ids", "labels", "label_mask", "visible"):
            left, right = a[key].get("support", {}), b[key].get("support", {})
            if left.get(field) != right.get(field):
                raise ValueError(f"different support {field} for {key}")
        if not query_control and a[key].get("readout_query") != b[key].get("readout_query"):
            raise ValueError(f"different readout query for {key}")


def paired_summary(differences, bootstrap=2000, seed=0):
    if not differences or bootstrap < 2:
        raise ValueError("paired bootstrap needs examples and at least two resamples")
    rng, n = random.Random(seed), len(differences)
    samples = sorted(sum(differences[rng.randrange(n)] for _ in range(n))/n
                     for _ in range(bootstrap))

    def quantile(q):
        position = (bootstrap-1)*q
        lo = int(position)
        hi = min(lo+1, bootstrap-1)
        return samples[lo] + (samples[hi]-samples[lo])*(position-lo)

    return {"difference": sum(differences)/n, "ci95": [quantile(.025), quantile(.975)], "n": n}


def analyze(additive_run, film_run, split="dev", bootstrap=2000, seed=0):
    roots = {"additive": Path(additive_run), "film": Path(film_run)}
    results = {mode: read_json(root/"result.json") for mode, root in roots.items()}
    for mode in roots:
        if results[mode].get("projector_fusion") != mode:
            raise ValueError(f"{mode} run has wrong projector_fusion metadata")
    fields = ("readout", "query_encoder_kind", "query_representation", "projector_query_mode",
              "projector_hidden", "projector_conditioning", "projector_cross_document",
              "projector_attention_dim", "projector_heads", "support_head", "support_head_input",
              "support_loss_weight", "support_warmup_steps", "support_visibility_policy",
              "max_query_len", "generator_lora_init", "train_decoder_input_mode", "offline_m")
    for field in fields:
        if results["additive"].get(field) != results["film"].get(field):
            raise ValueError(f"unmatched experiment metadata: {field}")
    normal = {mode: read_predictions(root/f"predictions_{split}_D0_Bfull.json")
              for mode, root in roots.items()}
    check_pair(normal["additive"], normal["film"])
    ids = list(normal["additive"])
    report = {"split": split, "units": "fraction", "bootstrap": bootstrap, "seed": seed,
              "qa": {}, "film_minus_additive": {}, "query_swap": None, "fusion_diagnostics": {}}
    for mode in roots:
        report["qa"][mode] = {metric: sum(normal[mode][i][metric] for i in ids)/len(ids)
                               for metric in METRICS}
        aggregate = results[mode]["metrics"][f"{split}|D0|B=full"]
        report["fusion_diagnostics"][mode] = {k: v for k, v in aggregate.items()
                                                if k.startswith("fusion_")}
    for metric in METRICS:
        report["film_minus_additive"][metric] = paired_summary(
            [normal["film"][i][metric]-normal["additive"][i][metric] for i in ids], bootstrap, seed)
    paths = {mode: root/f"predictions_{split}_mismatch-q_D0_Bfull.json" for mode, root in roots.items()}
    exists = [path.exists() for path in paths.values()]
    if any(exists) and not all(exists):
        raise ValueError("query-swap predictions must exist for both arms or neither")
    if all(exists):
        swapped = {mode: read_predictions(path) for mode, path in paths.items()}
        check_pair(swapped["additive"], swapped["film"])
        for mode in roots:
            check_pair(normal[mode], swapped[mode], query_control=True)
        drops = {mode: [normal[mode][i]["f1"]-swapped[mode][i]["f1"] for i in ids] for mode in roots}
        report["query_swap"] = {
            "normal_minus_mismatch_f1": {mode: paired_summary(values, bootstrap, seed)
                                          for mode, values in drops.items()},
            "film_minus_additive_drop": paired_summary(
                [f-a for f, a in zip(drops["film"], drops["additive"])], bootstrap, seed),
            "shares_answer_questions": {mode: sum(bool(swapped[mode][i].get("mismatch_shares_answer"))
                                                    for i in ids) for mode in roots},
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--additive_run", required=True)
    parser.add_argument("--film_run", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json")
    args = parser.parse_args()
    report = analyze(args.additive_run, args.film_run, args.split, args.bootstrap, args.seed)
    print("metric      additive     FiLM       FiLM - additive [95% CI], pp")
    for metric in METRICS:
        difference = report["film_minus_additive"][metric]
        lo, hi = difference["ci95"]
        print(f"{metric:<11} {100*report['qa']['additive'][metric]:8.2f} "
              f"{100*report['qa']['film'][metric]:8.2f} "
              f"{100*difference['difference']:+8.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
    print("Query-swap and pooled RMS diagnostics:")
    print(json.dumps({k: report[k] for k in ("query_swap", "fusion_diagnostics")}, indent=2, allow_nan=False))
    if args.output_json:
        destination = Path(args.output_json).resolve()
        sources = {path.resolve() for root in (Path(args.additive_run), Path(args.film_run))
                   for path in root.glob("*.json")}
        if destination in sources:
            raise ValueError("output_json must not overwrite an input run JSON file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                               encoding="utf-8")


if __name__ == "__main__":
    main()
