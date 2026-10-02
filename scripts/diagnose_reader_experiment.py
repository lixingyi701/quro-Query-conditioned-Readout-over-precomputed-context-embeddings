"""Read-only, fixed-sample evidence and state diagnostics for saved reader runs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from src.causal_order import paired_bootstrap
from src.data import move_to_device
from src.reader_experiment import StateTargets, cosine_state_loss, file_hash, row_key
from src.reader_runtime import (atomic_json, dataset, environment, evaluate, identity,
                                load_runtime, loader, seed_all)


def load_checkpoint(path, device):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("reader_format") != 1:
        raise ValueError("expected a reader checkpoint, including a saved step-zero checkpoint")
    args = argparse.Namespace(**saved["args"])
    args.device, args.grad_checkpointing = device, False
    seed_all(args.seed)
    cfg, cache, reader = load_runtime(args)
    current = identity(args, cfg, cache, reader)
    if any(saved["provenance"].get(k) != v for k, v in current.items()):
        raise ValueError("checkpoint base/data identity changed")
    reader.restore_state(saved["weights"])
    reader.eval()
    return saved, args, cfg, cache, reader, current


def mismatch_rows(rows):
    """A fixed permutation, independent of batching, with equal document counts.

    Reject overlapping documents so a supposed mismatch cannot silently retain
    the recipient's evidence. Keep question, golds and all other labels unchanged.
    A document count held by a single row has no possible donor; such rows are
    returned as excluded and must be dropped from every condition.
    """
    buckets = defaultdict(list)
    for i, row in enumerate(rows):
        buckets[len(row["retrieved_doc_ids"])].append(i)
    donors, excluded = {}, []
    for indices in buckets.values():
        if len(indices) == 1:
            excluded.append(rows[indices[0]]["id"])
            continue
        for shift in range(1, len(indices)):
            pairs = list(zip(indices, indices[shift:] + indices[:shift]))
            if all(not set(rows[i]["retrieved_doc_ids"]) & set(rows[j]["retrieved_doc_ids"])
                   for i, j in pairs):
                donors.update(pairs)
                break
        else:
            raise ValueError("cannot form disjoint equal-document-count donors; increase --limit")
    kept = sorted(donors)
    if not kept:
        raise ValueError("no row has an equal-document-count donor; increase --limit")
    result = [deepcopy(rows[i]) for i in kept]
    for row, i in zip(result, kept):
        row["retrieved_doc_ids"] = list(rows[donors[i]]["retrieved_doc_ids"])
    mapping = [{"id": rows[i]["id"], "donor_id": rows[donors[i]]["id"],
                "original_docs": rows[i]["retrieved_doc_ids"],
                "donor_docs": row["retrieved_doc_ids"]} for row, i in zip(result, kept)]
    return [rows[i] for i in kept], result, mapping, excluded


@contextmanager
def disabled_cross(reader):
    if reader.spec.arm != "w-ce":
        raise ValueError("disabled-cross control is W-only")
    saved = {k: block.gate.detach().clone() for k, block in reader.cross.items()}
    try:
        with torch.no_grad():
            for block in reader.cross.values():
                block.gate.zero_()
        yield
    finally:
        with torch.no_grad():
            for k, block in reader.cross.items():
                block.gate.copy_(saved[k])


def evidence(a, out):
    saved, original, cfg, cache, reader, current = load_checkpoint(a.checkpoint, a.device)
    ds = dataset(a.eval_file, cfg, cache, reader, a.limit)
    ds.rows, changed, mapping, excluded = mismatch_rows(ds.rows)
    atomic_json(out / "donors.json", {"excluded_without_donor": excluded, "pairs": mapping})
    conditions = {}
    def run(name):
        summary, rows = evaluate(reader, loader(ds, cfg, cache, reader, a.batch_size),
                                 a.device, original.max_new_tokens)
        atomic_json(out / f"{name}_predictions.json", rows)
        conditions[name] = {"summary": summary, "rows": rows}
    run("correct")
    original_rows = ds.rows
    try:
        ds.rows = changed
        run("mismatch")
    finally:
        ds.rows = original_rows
    if reader.spec.arm == "w-ce":
        with disabled_cross(reader):
            run("disabled_cross")
    contrasts = {}
    for name, result in conditions.items():
        if name == "correct":
            continue
        contrasts[name] = {}
        for split in ("all", "bridge", "comparison"):
            pairs = [(r, c) for r, c in zip(conditions["correct"]["rows"], result["rows"])
                     if split == "all" or r["hop_type"] == split]
            if pairs:
                contrasts[name][split] = {
                    **{metric: paired_bootstrap([c[metric] for r, c in pairs],
                                               [r[metric] for r, c in pairs], seed=42)
                       for metric in ("substring", "em", "f1", "nll")},
                    "prediction_change_rate": sum(r["pred"] != c["pred"] for r, c in pairs) / len(pairs)}
    atomic_json(out / "summary.json", {
        "checkpoint": str(Path(a.checkpoint).resolve()), "checkpoint_sha256": file_hash(a.checkpoint),
        "step": saved["step"], "arm": original.arm, "provenance": current,
        "eval_file_sha256": file_hash(a.eval_file), "args": vars(a), "environment": environment(),
        "n_evaluated": len(ds.rows), "excluded_without_donor": excluded,
        "conditions": {k: v["summary"] for k, v in conditions.items()}, "contrasts": contrasts,
        "delta_direction": "intervention minus correct; positive NLL / negative QA indicates useful evidence",
        "scope": "exploratory; CI conditional on this fixed donor permutation, not training-seed uncertainty"})


def gradient_stats(ce, state, params, weight):
    g_ce = torch.autograd.grad(ce, params, retain_graph=True, allow_unused=True)
    g_state = torch.autograd.grad(state, params, allow_unused=True)
    # Avoid concatenating potentially large LoRA gradient vectors.
    ce2 = sum(float(g.float().square().sum()) for g in g_ce if g is not None)
    st2 = sum(float(g.float().square().sum()) for g in g_state if g is not None)
    dot = sum(float(x.float().mul(y.float()).sum()) for x, y in zip(g_ce, g_state)
              if x is not None and y is not None)
    nc, ns = ce2 ** .5, st2 ** .5
    return {"ce_grad_norm": nc, "state_grad_norm": ns,
            "weighted_state_grad_norm": weight * ns,
            "weighted_state_to_ce_ratio": weight * ns / nc if nc else None,
            "gradient_cosine": dot / (nc * ns) if nc and ns else None}


def state(a, out):
    first, original, cfg, cache, reader, current = load_checkpoint(a.checkpoints[0], a.device)
    if original.arm == "w-ce":
        raise ValueError("state diagnostic currently compares Direct checkpoints only")
    ds = dataset(cfg.data.train_file, cfg, cache, reader, a.limit)
    targets = StateTargets(a.teacher_cache, current)
    if any(row_key(r) not in targets.index for r in ds.rows):
        raise ValueError("teacher targets missing fixed samples")
    atomic_json(out / "samples.json", [{"id": r["id"], "row_key": row_key(r)} for r in ds.rows])
    params = [p for p in reader.lm.parameters() if p.requires_grad]
    reports = []
    for path in a.checkpoints:
        saved = first if path == a.checkpoints[0] else torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("reader_format") != 1 or saved["args"]["arm"] == "w-ce":
            raise ValueError("only compatible Direct reader checkpoints accepted")
        if any(saved["provenance"].get(k) != v for k, v in current.items()):
            raise ValueError("checkpoint provenance differs")
        reader.restore_state(saved["weights"])
        weight = saved["args"].get("state_weight", .1)
        records, gradients = [], []
        for index, batch in enumerate(loader(ds, cfg, cache, reader, a.batch_size)):
            batch = move_to_device(batch, a.device)
            teacher = targets.gather(batch["raw"], a.device).float()
            # Prompt only: no answer tokens, no dropout, same examples at every checkpoint.
            with torch.no_grad():
                packed = reader.pack(batch, targets=False)
                with reader.activate(packed, capture=True) as captured:
                    reader.lm.get_decoder()(**packed["inputs"], use_cache=False)
                    student = reader.stack_states(captured).float()
                distances = 1 - F.cosine_similarity(student, teacher, dim=-1)
                relative = (student - teacher).norm(dim=-1) / teacher.norm(dim=-1).clamp_min(1e-8)
                for i, row in enumerate(batch["raw"]):
                    records.append({"id": row["id"], "cosine_distance": distances[i].tolist(),
                                    "relative_l2": relative[i].tolist()})
            if index < a.grad_batches:
                packed = reader.pack(batch, targets=True)
                with reader.activate(packed, capture=True) as captured:
                    result = reader.lm(**packed["inputs"], use_cache=False)
                    loss = cosine_state_loss(reader.stack_states(captured), teacher)
                    gradients.append({"ids": batch["ids"], "ce": float(result.loss.detach()),
                                      "state": float(loss.detach()),
                                      **gradient_stats(result.loss, loss, params, weight)})
                del result, loss, captured, packed
        report = {"checkpoint": str(Path(path).resolve()), "checkpoint_sha256": file_hash(path),
                  "step": saved["step"], "state_weight": weight, "rows": records,
                  "mean_cosine_distance_by_layer": torch.tensor([r["cosine_distance"] for r in records]).mean(0).tolist(),
                  "mean_relative_l2_by_layer": torch.tensor([r["relative_l2"] for r in records]).mean(0).tolist(),
                  "gradients": gradients}
        reports.append(report)
        atomic_json(out / "summary.json", {"args": vars(a), "provenance": current,
                    "teacher_manifest_sha256": file_hash(Path(a.teacher_cache) / "manifest.json"),
                    "layers": list(reader.spec.layers), "environment": environment(), "reports": reports,
                    "scope": "train-prefix diagnostic, not held-out generalization; gradients in eval mode, gold CE, no optimizer step"})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("evidence", "state"):
        s = sub.add_parser(command)
        s.add_argument("--out_dir", required=True)
        s.add_argument("--device", default="cuda")
        s.add_argument("--batch_size", type=int, default=1)
        s.add_argument("--limit", type=int, default=128 if command == "state" else 500)
        if command == "evidence":
            s.add_argument("--checkpoint", required=True)
            s.add_argument("--eval_file", required=True)
        else:
            s.add_argument("--checkpoints", nargs="+", required=True)
            s.add_argument("--teacher_cache", required=True)
            s.add_argument("--grad_batches", type=int, default=4)
    return p


def main(a):
    if a.limit < 1 or a.batch_size < 1 or getattr(a, "grad_batches", 0) < 0:
        raise ValueError("invalid sample/batch limits")
    out = Path(a.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing to overwrite a nonempty output directory")
    out.mkdir(parents=True, exist_ok=True)
    {"evidence": evidence, "state": state}[a.command](a, out)
    print(f"[diagnostic complete] {out}")


if __name__ == "__main__":
    main(parser().parse_args())
