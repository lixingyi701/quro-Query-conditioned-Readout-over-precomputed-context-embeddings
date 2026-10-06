"""Recompute evidence coverage from saved predictions, without running a model.

    python scripts/analyze_support_topk.py --predictions normal.json \
        --mismatch_predictions mismatch-q.json --topk 2,4,6 --output_json topk.json

Fractions in JSON, percentages in the table. Token counts are hypothetical
selected-memory budgets, not measured latency or QA after selection.
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.support_metrics import rank_support, support_scores


def parse_topks(value):
    try:
        ks = sorted({int(part) for part in value.split(",")})
        if not ks or min(ks) < 1:
            raise ValueError()
        return ks
    except ValueError as error:
        raise argparse.ArgumentTypeError("topk must be comma-separated positive integers") from error


def load_predictions(path):
    with open(path, encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected the predictions JSON array")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError(f"{path}: prediction row has no id")
        if row["id"] in result:
            raise ValueError(f"{path}: duplicate question id {row['id']}")
        result[row["id"]] = row
    return result


def evidence(row, topks):
    support = row.get("support")
    if support is None:
        return None
    logits, labels = support["logits"], support["labels"]
    mask = support.get("label_mask", [True]*len(labels))
    ranked, gold = rank_support(logits, labels, mask)
    if not gold:
        return None
    visible = support.get("visible", [None]*len(labels))
    if len(visible) != len(labels):
        raise ValueError(f"{row['id']}: support visibility length differs from labels")
    doc_ids = support.get("doc_ids")
    if doc_ids is not None and len(doc_ids) != len(labels):
        raise ValueError(f"{row['id']}: document ids differ in length from labels")
    return {"metrics": support_scores(logits, labels, mask, topks),
            "selected": {k: set(ranked[:k]) for k in topks},
            "valid_documents": len(ranked), "gold_documents": len(gold),
            "all_gold_visible": all(visible[i] is True for i in gold)}


def average(values):
    values = [v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def aggregate(items, topks, memories_per_document):
    result = {"questions": len(items),
              "both_questions": sum(x["gold_documents"] == 2 for x in items),
              "mean_valid_documents": average([x["valid_documents"] for x in items]),
              "topk": {}}
    for k in topks:
        selected = [min(k, x["valid_documents"]) for x in items]
        result["topk"][str(k)] = {
            "recall": average([x["metrics"][f"recall_at_{k}"] for x in items]),
            "both": average([x["metrics"][f"both_at_{k}"] for x in items]),
            "mean_selected_documents": average(selected),
            "mean_selected_memory_tokens": average([n*memories_per_document for n in selected]),
        }
    return result


def paired_stat(differences, resamples, seed):
    result = {"difference": average(differences), "questions": len(differences), "ci95": None}
    if differences and resamples:
        rnd, n = random.Random(seed), len(differences)
        means = sorted(sum(differences[rnd.randrange(n)] for _ in range(n))/n
                       for _ in range(resamples))
        result["ci95"] = [means[math.floor(.025*resamples)],
                          means[max(0, math.ceil(.975*resamples)-1)]]
    return result


def analyze(normal, mismatch=None, topks=(2, 4, 6), memories_per_document=8,
            resamples=2000, seed=0):
    if memories_per_document < 1 or resamples < 0:
        raise ValueError("memory count must be positive and bootstrap count nonnegative")
    topks = tuple(topks)
    if not topks or any(type(k) is not int or k < 1 for k in topks):
        raise ValueError("top-k values must be positive integers")
    items = {i: x for i, row in normal.items() if (x := evidence(row, topks)) is not None}
    result = {"prediction_questions": len(normal), "skipped_questions": len(normal)-len(items),
              "memories_per_document": memories_per_document,
              "bootstrap_resamples": resamples, "bootstrap_seed": seed,
              "normal": {}}
    groups = {"all": list(items),
              "all_gold_visible": [i for i, x in items.items() if x["all_gold_visible"]]}
    for group, ids in groups.items():
        result["normal"][group] = aggregate([items[i] for i in ids], topks, memories_per_document)
    if mismatch is None:
        return result
    if set(normal) != set(mismatch):
        raise ValueError("normal/mismatch prediction id sets differ; paired analysis requires the same questions")
    for i, row in normal.items():
        a, b = row.get("support"), mismatch[i].get("support")
        if (a is None) != (b is None):
            raise ValueError(f"{i}: support metadata missing from only one condition")
        if a is None:
            continue
        for key in ("labels", "label_mask", "visible"):
            default = ([True]*len(a["labels"]) if key == "label_mask"
                       else [None]*len(a["labels"]) if key == "visible" else None)
            if a.get(key, default) != b.get(key, default):
                raise ValueError(f"{i}: {key} differs; expected query-only mismatch with original targets")
        if ("doc_ids" in a and "doc_ids" in b) and a["doc_ids"] != b["doc_ids"]:
            raise ValueError(f"{i}: document order differs between normal and query mismatch")
    other = {i: x for i, row in mismatch.items() if (x := evidence(row, topks)) is not None}
    if set(items) != set(other):
        raise ValueError("normal/mismatch labelled question sets differ")
    result["mismatch"] = {}
    result["query_swap"] = {}
    result["mismatch_shares_answer_questions"] = sum(
        bool(mismatch[i].get("mismatch_shares_answer")) for i in items)
    result["document_order_verified_questions"] = sum(
        "doc_ids" in normal[i]["support"] and "doc_ids" in mismatch[i]["support"] for i in items)
    for group, ids in groups.items():
        result["mismatch"][group] = aggregate([other[i] for i in ids], topks, memories_per_document)
        result["query_swap"][group] = {}
        for k in topks:
            stats = {"same_selected_set_fraction": average([
                float(items[i]["selected"][k] == other[i]["selected"][k]) for i in ids])}
            for metric in ("recall", "both"):
                key = f"{metric}_at_{k}"
                differences = [other[i]["metrics"][key]-items[i]["metrics"][key]
                               for i in ids if items[i]["metrics"][key] is not None]
                stats[f"{metric}_mismatch_minus_normal"] = paired_stat(differences, resamples, seed)
            result["query_swap"][group][str(k)] = stats
    return result


def display(result):
    def percent(value):
        return "—" if value is None else f"{100*value:.2f}"

    print("Evidence coverage only; memory budgets are hypothetical, not measured speed or selected QA.")
    print(f"Questions: {result['prediction_questions']}; skipped: {result['skipped_questions']}")
    for condition in ("normal", "mismatch"):
        for group, table in result.get(condition, {}).items():
            print(f"\n{condition}/{group}: n={table['questions']}, both denominator={table['both_questions']}")
            print("k\tRecall (%)\tboth (%)\tmean memory tokens\tsame set after query swap (%)")
            for k, row in table["topk"].items():
                same = result.get("query_swap", {}).get(group, {}).get(k, {}).get("same_selected_set_fraction")
                print(f"{k}\t{percent(row['recall'])}\t{percent(row['both'])}\t"
                      f"{row['mean_selected_memory_tokens']}\t{percent(same)}")
    if "query_swap" in result:
        print("\nPaired differences are mismatch minus normal (percentage points):")
        for group, ks in result["query_swap"].items():
            for k, row in ks.items():
                for metric in ("recall", "both"):
                    x = row[f"{metric}_mismatch_minus_normal"]
                    interval = [100*v for v in x["ci95"]] if x["ci95"] is not None else None
                    delta = 100*x["difference"] if x["difference"] is not None else None
                    print(f"{group}/k={k}/{metric}: {delta}, CI95={interval}, n={x['questions']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--mismatch_predictions")
    parser.add_argument("--topk", type=parse_topks, default=[2, 4, 6])
    parser.add_argument("--memories_per_document", type=int, default=8)
    parser.add_argument("--bootstrap_resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json")
    args = parser.parse_args()
    report = analyze(load_predictions(args.predictions),
                     load_predictions(args.mismatch_predictions) if args.mismatch_predictions else None,
                     args.topk, args.memories_per_document, args.bootstrap_resamples, args.seed)
    report["sources"] = {"normal": args.predictions, "mismatch": args.mismatch_predictions}
    display(report)
    if args.output_json:
        target = Path(args.output_json)
        if target.resolve() in {Path(p).resolve() for p in report["sources"].values() if p}:
            raise ValueError("output must not overwrite input predictions")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
