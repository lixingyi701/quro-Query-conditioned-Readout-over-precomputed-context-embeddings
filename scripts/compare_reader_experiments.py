"""Paired QA/NLL comparisons from run_reader_experiment.py eval directories."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.causal_order import paired_bootstrap
from src.reader_runtime import atomic_json


def compare(reference, candidate):
    def read(root):
        root = Path(root)
        meta = json.loads((root / "summary.json").read_text())
        rows = json.loads((root / "predictions.json").read_text())
        index = {str(row["id"]): row for row in rows}
        if len(index) != len(rows):
            raise ValueError("duplicate prediction IDs")
        return meta, index
    ref_meta, ref = read(reference)
    cand_meta, cand = read(candidate)
    for field in ("eval_file_sha256", "gold_only", "conditions"):
        if ref_meta[field] != cand_meta[field]:
            raise ValueError(f"evaluation conditions differ: {field}")
    if ref.keys() != cand.keys():
        raise ValueError("prediction sample sets differ; do not silently take an intersection")
    for key in ref:
        if ref[key]["golds"] != cand[key]["golds"] or ref[key]["hop_type"] != cand[key]["hop_type"]:
            raise ValueError(f"reference labels differ for {key}")
    output = {"reference": str(reference), "candidate": str(candidate),
              "delta_direction": "candidate minus reference; negative NLL is better", "contrasts": {}}
    for subset in ("all", "bridge", "comparison"):
        ids = sorted(k for k in ref if subset == "all" or ref[k]["hop_type"] == subset)
        if ids:
            output["contrasts"][subset] = {m: paired_bootstrap(
                [cand[k][m] for k in ids], [ref[k][m] for k in ids], seed=42)
                for m in ("substring", "em", "f1", "nll")}
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", nargs="+", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if Path(a.output).exists():
        raise FileExistsError(a.output)
    results = [compare(a.reference, c) for c in a.candidate]
    atomic_json(a.output, results)
    for result in results:
        print(result["candidate"])
        for subset, values in result["contrasts"].items():
            value = values["substring"]
            print(f"  {subset}: {value['delta']*100:+.2f} pp "
                  f"[{value['lo']*100:+.2f}, {value['hi']*100:+.2f}], n={value['n']}")
