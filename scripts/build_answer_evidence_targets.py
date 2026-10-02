"""Audit frozen published PISCO teacher views; optionally export TRAIN SKD labels.

No parameter updates. Audit does not export trainable labels. Export covers the
entire supplied train file and refuses dev/test/tune overlap by ID or question.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.nn import functional as F

from scripts.prepare_reader_reset import keys, read
from scripts.run_reader_experiment import make_output
from src.answer_evidence import export_variants, roles, teacher_view
from src.data import load_corpus
from src.metrics import aggregate, score
from src.prompt import assemble_inputs
from src.reader_experiment import file_hash, row_key
from src.reader_runtime import atomic_json, dataset, environment, identity, load_runtime, seed_all
from src.causal_order import paired_bootstrap


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["audit", "export"], default="audit")
    p.add_argument("--generator_path", required=True)
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--train_file", required=True)
    p.add_argument("--exclude_files", nargs="+", required=True, help="ALL tune/dev/final-test query files")
    p.add_argument("--corpus", nargs="+", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--preset", default="pisco_hotpot")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
    p.add_argument("--max_docs", type=int, default=10)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--limit", type=int, help="audit only; default audit sample=512")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--layers", default="8,16,24", help="runtime metadata only; no hidden-state target")
    p.set_defaults(init_source="published", init_checkpoint=None, arm="direct-ce", state_weight=0.,
                   workspace_tokens=16, cross_dim=128, cross_heads=8, gate_init=.1,
                   grad_checkpointing=False, freeze_decoder=True)
    return p


@torch.no_grad()
def evaluate_view(reader, cache, entries, mode, corpus, batch_size, max_new):
    """Answer-relative gold NLL (including EOS) + independent free generation."""
    result = {}
    builder = reader.builders["D0"]
    device = next(reader.lm.parameters()).device
    for start in range(0, len(entries), batch_size):
        chunk = entries[start:start+batch_size]
        views = [teacher_view(r, builder, corpus, mode, role) for r, role in chunk]
        prompts = [p for p, _ in views]
        # One filler row permits an entirely raw batch; no filler is inserted into
        # the prompt because there are zero slot positions.
        hidden = reader.hidden
        width = max(1, max(len(ids) * builder.n_mem_tokens for _, ids in views))
        soft = torch.zeros(len(chunk), width, hidden, device=device)
        mask = torch.zeros(len(chunk), width, dtype=torch.bool, device=device)
        for i, (_, ids) in enumerate(views):
            if ids:
                z, _, _ = cache.get_many([ids], device=device)
                z = z.flatten(1, 2)[0]
                soft[i, :len(z)] = z
                mask[i, :len(z)] = True
        targets = [reader.tok(" " + r["answers"][0].strip(), add_special_tokens=False)["input_ids"][:reader.max_answer_len]
                   + [reader.tok.eos_token_id] for r, _ in chunk]
        packed = assemble_inputs(reader.lm.get_input_embeddings(), prompts, soft, mask, targets,
                                 pad_token_id=reader.pad_id)
        labels = packed["labels"][:, 1:]
        logits = reader.lm(**packed, use_cache=False).logits[:, :-1].float()
        losses = F.cross_entropy(logits.transpose(1, 2), labels.clamp_min(0), reduction="none")
        token_nll = [losses[i][labels[i] != -100].cpu().tolist() for i in range(len(chunk))]
        if not all(torch.isfinite(torch.tensor(v)).all().item() for v in token_nll):
            raise ValueError("nonfinite teacher NLL")
        del logits, losses, packed
        packed = assemble_inputs(reader.lm.get_input_embeddings(), prompts, soft, mask, None,
                                 pad_token_id=reader.pad_id, pad_side="left")
        generated = reader.lm.generate(**packed, max_new_tokens=max_new, do_sample=False,
                                       use_cache=True, eos_token_id=reader.tok.eos_token_id,
                                       pad_token_id=reader.pad_id)
        for (row, role), ids, values, tgt, prompt in zip(chunk, generated, token_nll, targets, prompts):
            tokens = ids.cpu().tolist()
            pred = reader.tok.decode(tokens, skip_special_tokens=True).strip()
            target_ids = reader.tok(" " + pred, add_special_tokens=False)["input_ids"]
            result[row_key(row)] = {"pred": pred, **score(pred, row["answers"]),
                "nll": sum(values)/len(values), "gold_token_ids": tgt, "gold_token_nll": values,
                "answer_content_nll": sum(values[:-1])/len(values[:-1]) if len(values) > 1 else None,
                "target_length": len(target_ids), "target_token_ids": target_ids,
                "stopped_eos": reader.tok.eos_token_id in tokens,
                "prompt_tokens": len(prompt.input_ids)}
    return result


def main(a):
    if min(a.batch_size, a.max_docs, a.max_new_tokens) < 1 or (a.limit is not None and a.limit < 1):
        raise ValueError("positive batch/token/sample limits required")
    if a.mode == "export" and a.limit is not None:
        raise ValueError("export must cover its entire train file; --limit is audit-only")
    source = read(a.train_file)
    blocked = [r for path in a.exclude_files for r in read(path)]
    ids, qs = keys(source)
    bi, bq = keys(blocked)
    if ids & bi or qs & bq:
        raise ValueError("training overlaps supplied tune/dev/test by ID or normalized question")
    out = make_output(a.out_dir)
    seed_all(a.seed)
    cfg, cache, reader = load_runtime(a, teacher=True)
    reader.max_answer_len = cfg.data.max_answer_len
    provenance = identity(a, cfg, cache, reader)
    corpus = load_corpus(a.corpus)
    ds = dataset(a.train_file, cfg, cache, reader, corpus=corpus)
    rows = list(ds.rows)
    if a.mode == "audit":
        rows.sort(key=row_key)
        random.Random(a.seed).shuffle(rows)
        rows = rows[:a.limit or 512]
    inputs = {str(p): file_hash(p) for p in [a.train_file, *a.exclude_files, *a.corpus]}
    atomic_json(out/"manifest.json", {"complete": False, "mode": a.mode, "args": vars(a)})
    reports, reasons, streams = [], Counter(), {}
    names = ("gold", "raw_skd_all", "raw_skd_matched", "answer_skd_matched")
    started = time.perf_counter()
    try:
        if a.mode == "export":
            streams = {n: open(out/f"{n}.jsonl", "w", encoding="utf-8") for n in names}
        with open(out/"teacher_audit.jsonl", "w", encoding="utf-8") as audit:
            for start in range(0, len(rows), a.batch_size):
                chunk = rows[start:start+a.batch_size]
                annotated = [(row, *roles(row, reader.builders["D0"], corpus)) for row in chunk]
                all_entries = [(r, role) for r, role, reason in annotated]
                eligible = [(r, role) for r, role, reason in annotated if role is not None]
                values = {m: evaluate_view(reader, cache, eligible if m in {"A", "B"} else all_entries,
                                         m, corpus, a.batch_size, a.max_new_tokens) for m in ("M", "R", "A", "B")}
                for row, role, reason in annotated:
                    conditions = {m: v[row_key(row)] for m, v in values.items() if row_key(row) in v}
                    variants, flags = export_variants(row, conditions, cfg.data.max_answer_len)
                    flags["raw_target_tokens_equal_gold"] = bool(flags["raw_eligible"] and
                        conditions["R"]["target_token_ids"] == conditions["R"]["gold_token_ids"][:-1])
                    flags["answer_target_tokens_equal_gold"] = bool(flags["matched_eligible"] and
                        conditions["A"]["target_token_ids"] == conditions["A"]["gold_token_ids"][:-1])
                    flags["matched_target_tokens_identical"] = bool(flags["matched_eligible"] and
                        conditions["A"]["target_token_ids"] == conditions["R"]["target_token_ids"])
                    rec = {"id": row["id"], "query": row["query"], "golds": row["answers"],
                           "row_key": row_key(row), "role": role, "role_exclusion": reason,
                           "conditions": conditions, **flags}
                    reports.append(rec)
                    reasons[reason or "role_defined"] += 1
                    audit.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    for name, stream in streams.items():
                        stream.write(json.dumps(variants[name], ensure_ascii=False) + "\n")
                audit.flush()
                print(f"[{a.mode}] {min(start+a.batch_size,len(rows))}/{len(rows)}", flush=True)
    finally:
        for stream in streams.values():
            stream.close()
    metrics = {}
    for mode in ("M", "R", "A", "B"):
        selected = [r["conditions"][mode] for r in reports if mode in r["conditions"]]
        metrics[mode] = aggregate(selected)
        if selected:
            metrics[mode]["nll"] = sum(r["nll"] for r in selected)/len(selected)
    contrasts = {}
    for mode, reference in (("R", "M"), ("A", "M"), ("B", "M"), ("A", "B"), ("A", "R")):
        paired = [r["conditions"] for r in reports if mode in r["conditions"] and reference in r["conditions"]]
        if paired:
            name = mode+"-"+reference
            contrasts[name] = {k: paired_bootstrap([r[mode][k] for r in paired],
                        [r[reference][k] for r in paired], seed=a.seed) for k in ("substring", "em", "f1", "nll")}
            content = [r for r in paired if r[mode]["answer_content_nll"] is not None and
                       r[reference]["answer_content_nll"] is not None]
            if content:
                contrasts[name]["answer_content_nll"] = paired_bootstrap(
                    [r[mode]["answer_content_nll"] for r in content],
                    [r[reference]["answer_content_nll"] for r in content], seed=a.seed)
            contrasts[name].update(repaired=sum(r[mode]["substring"] > r[reference]["substring"] for r in paired),
                                   broken=sum(r[mode]["substring"] < r[reference]["substring"] for r in paired))
    counters = {k: sum(bool(r[k]) for r in reports) for k in (
        "raw_eligible","matched_eligible","matched_targets_identical","raw_target_equals_gold","answer_target_equals_gold",
        "raw_target_tokens_equal_gold", "answer_target_tokens_equal_gold", "matched_target_tokens_identical")}
    counters["raw_targets_token_different_from_gold"] = counters["raw_eligible"] - counters["raw_target_tokens_equal_gold"]
    counters["matched_targets_token_different"] = counters["matched_eligible"] - counters["matched_target_tokens_identical"]
    summary = {"n": len(reports), "role_counts": dict(reasons), "metrics": metrics,
        "contrasts": contrasts, "target_counts": counters,
        "paired_scope": "A/B contrasts use only role-defined bridge rows, not all M rows",
        "decision": "AUDIT_ONLY_NO_AUTOMATIC_TRAINING",
        "limits": "Training queries only. Substring screening does not verify reasoning. Raw-view improvement "
                  "does not prove that its information survived in Z. SKD is not a novelty claim."}
    atomic_json(out/"summary.json", summary)
    outputs = {n: {"file": f"{n}.jsonl", "sha256": file_hash(out/f"{n}.jsonl")} for n in streams}
    atomic_json(out/"manifest.json", {"complete": True, "format": "answer_evidence_skd_v1", "mode": a.mode,
        "args": vars(a), "provenance": provenance, "input_sha256": inputs, "outputs": outputs,
        "rows": len(rows), "max_answer_len": cfg.data.max_answer_len, "target_counts": counters,
        "audit_sha256": file_hash(out/"teacher_audit.jsonl"), "environment": environment(),
        "seconds": time.perf_counter()-started})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(parser().parse_args())
