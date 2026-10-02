"""Fail a reset pilot if step-zero predictions/NLL or provenance differ."""
import argparse
import json
import math
from pathlib import Path


def check(reference, candidates, atol=1e-6):
    def read(root):
        root = Path(root)
        meta = json.loads((root / "manifest.json").read_text())
        rows = json.loads((root / "step_zero_predictions.json").read_text())
        index = {str(r["id"]): r for r in rows}
        if not rows or len(index) != len(rows):
            raise ValueError("empty or duplicate IDs")
        return meta, index
    ref, rows = read(reference)
    if atol < 0:
        raise ValueError("negative tolerance")
    for path in candidates:
        meta, candidate = read(path)
        if meta["provenance"] != ref["provenance"] or candidate.keys() != rows.keys():
            raise ValueError(f"{path}: identity/sample mismatch")
        for field in ("eval_samples", "eval_batch_size", "max_new_tokens", "attn_implementation"):
            if meta["args"][field] != ref["args"][field]:
                raise ValueError(f"{path}: evaluation setting differs: {field}")
        for key, row in rows.items():
            other = candidate[key]
            if row["golds"] != other["golds"] or row["pred"] != other["pred"]:
                raise ValueError(f"{path}: prediction/gold mismatch at {key}")
            if not math.isfinite(row["nll"]) or not math.isfinite(other["nll"]) or abs(row["nll"]-other["nll"]) > atol:
                raise ValueError(f"{path}: NLL mismatch at {key}")
    print(f"PASS: {len(rows)} examples; step-zero outputs/NLL match in {len(candidates)} candidates. "
          "This checks reported behavior, not every vocabulary logit or upstream release authenticity.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--nll_atol", type=float, default=1e-6)
    a = p.parse_args()
    check(a.reference, a.candidates, a.nll_atol)
