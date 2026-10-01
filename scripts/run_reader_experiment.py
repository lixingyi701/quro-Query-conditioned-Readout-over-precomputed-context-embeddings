"""CLI for train-only raw targets and matched Direct-CE / Direct-State / W-CE.

See docs/READER_STATE_WORKSPACE_RUNBOOK.md. Single process, one visible GPU.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from src.data import load_corpus, move_to_device
from src.reader_experiment import StateTargets, cosine_state_loss, file_hash, row_key
from src.reader_runtime import (EpochBatches, atomic_json, dataset, environment, evaluate,
    identity, load_runtime, loader, restore_rng, save_checkpoint, seed_all, training_signature)
from src.train import lr_lambda_factory


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    sub = top.add_subparsers(dest="command", required=True)
    for command in ("cache", "train"):
        p = sub.add_parser(command)
        p.add_argument("--init_checkpoint", required=True, help="P1 checkpoint; weights-only initialization")
        p.add_argument("--preset", default="pisco_hotpot")
        p.add_argument("--generator_path")
        p.add_argument("--cache_dir")
        p.add_argument("--train_file")
        p.add_argument("--device", default="cuda")
        p.add_argument("--attn_implementation", choices=["eager", "sdpa"], default="sdpa")
        p.add_argument("--max_docs", type=int, default=10)
        p.add_argument("--layers", default="8,16,24", help="one-based block outputs")
        p.add_argument("--batch_size", type=int, default=4 if command == "cache" else 2)
        p.add_argument("--num_workers", type=int, default=0)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--limit_train", type=int)
        p.add_argument("--out_dir", required=True)
        p.set_defaults(arm="direct-ce", state_weight=0.1, workspace_tokens=16,
                       cross_dim=512, cross_heads=8, gate_init=0.1, grad_checkpointing=False)
        if command == "cache":
            p.add_argument("--corpus", nargs="+", required=True)
        else:
            p.add_argument("--arm", choices=["direct-ce", "direct-state", "w-ce"], required=True)
            p.add_argument("--state_weight", type=float, default=0.1)
            p.add_argument("--teacher_cache")
            p.add_argument("--train_target", choices=["gold", "p1"], default="gold",
                           help="p1 inherits the checkpoint's teacher-output preference; eval always uses gold")
            p.add_argument("--workspace_tokens", type=int, default=16)
            p.add_argument("--cross_dim", type=int, default=512)
            p.add_argument("--cross_heads", type=int, default=8)
            p.add_argument("--gate_init", type=float, default=0.1)
            p.add_argument("--steps", type=int, default=3000)
            p.add_argument("--grad_accum", type=int, default=8)
            p.add_argument("--decoder_lr", type=float, default=1e-4)
            p.add_argument("--workspace_lr", type=float, default=1e-4)
            p.add_argument("--warmup_ratio", type=float, default=0.05)
            p.add_argument("--weight_decay", type=float, default=0.01)
            p.add_argument("--grad_clip", type=float, default=1.0)
            p.add_argument("--grad_checkpointing", action="store_true")
            p.add_argument("--eval_file", required=True, help="dev only during method selection")
            p.add_argument("--eval_every", type=int, default=250)
            p.add_argument("--eval_samples", type=int, default=500)
            p.add_argument("--eval_batch_size", type=int, default=8)
            p.add_argument("--max_new_tokens", type=int, default=32)
            p.add_argument("--select_metric", choices=["substring", "em", "f1"], default="substring")
            p.add_argument("--save_every", type=int, default=250)
            p.add_argument("--log_every", type=int, default=20)
            p.add_argument("--resume", help="resume an interrupted reader run, with all original arguments")
    p = sub.add_parser("eval")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--limit", type=int)
    p.add_argument("--gold_only", action="store_true", help="K=2 gold evaluation, original retrieval order")
    return top


def make_output(path, resume=False):
    out = Path(path)
    if out.exists() and any(out.iterdir()) and not resume:
        raise ValueError(f"refusing to overwrite nonempty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def build_targets(a):
    out = make_output(a.out_dir)
    seed_all(a.seed)
    cfg, cache, reader = load_runtime(a, teacher=True)
    provenance = identity(a, cfg, cache, reader)
    corpus = load_corpus(a.corpus)
    ds = dataset(cfg.data.train_file, cfg, cache, reader, a.limit_train, corpus)
    batches = loader(ds, cfg, cache, reader, a.batch_size, a.num_workers)
    shape = (len(ds), len(reader.spec.layers), reader.hidden)
    meta = {**provenance, "shape": list(shape), "split": "train", "complete": False,
            "teacher": "frozen P1 raw", "contains_answer": False,
            "corpus_sha256": {str(Path(p).resolve()): file_hash(p) for p in a.corpus},
            "environment": environment(), "args": vars(a)}
    atomic_json(out / "manifest.json", meta)
    values = np.memmap(out / "states.bin", mode="w+", dtype="float16", shape=shape)
    index, offset = {}, 0
    started = time.perf_counter()
    with torch.no_grad():
        for batch in batches:
            batch = move_to_device(batch, a.device)
            packed = reader.pack(batch, targets=False, raw=True)
            assert "labels" not in packed["inputs"]
            with reader.activate(packed, capture=True) as captured:
                # Only the backbone is needed; do not materialize raw prompt vocabulary logits.
                reader.lm.get_decoder()(**packed["inputs"], use_cache=False)
                states = reader.stack_states(captured).float()
            states = states.cpu().half()
            if not torch.isfinite(states).all():
                raise ValueError("non-finite/FP16-overflow teacher target")
            n = states.size(0)
            values[offset:offset + n] = states.numpy()
            for i, row in enumerate(batch["raw"]):
                key = row_key(row)
                if key in index:
                    raise ValueError("duplicate teacher row key")
                index[key] = offset + i
            offset += n
            if offset % 1000 < n:
                print(f"[teacher] {offset}/{len(ds)}", flush=True)
    values.flush()
    atomic_json(out / "index.json", index)
    meta.update(complete=True, build_seconds=time.perf_counter() - started,
                index_sha256=file_hash(out / "index.json"),
                states_sha256=file_hash(out / "states.bin"))
    atomic_json(out / "manifest.json", meta)
    print(f"[teacher complete] {out}")


def train(a):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("single-process runner: launch one arm per GPU; do not use torchrun")
    if a.steps < 0 or min(a.grad_accum, a.eval_every, a.save_every, a.log_every) < 1:
        raise ValueError("invalid steps / interval")
    if min(a.batch_size, a.eval_batch_size, a.eval_samples, a.max_docs, a.max_new_tokens) < 1:
        raise ValueError("batch sizes, evaluation size and context limits must be positive")
    if min(a.decoder_lr, a.workspace_lr, a.grad_clip) <= 0 or not 0 <= a.warmup_ratio <= 1:
        raise ValueError("invalid optimizer settings")
    if a.limit_train is not None and a.limit_train < 1:
        raise ValueError("limit_train must be positive")
    if a.teacher_cache and a.arm != "direct-state":
        raise ValueError("only Direct-State may load teacher targets")
    out = make_output(a.out_dir, bool(a.resume))
    seed_all(a.seed)
    cfg, cache, reader = load_runtime(a)
    provenance = identity(a, cfg, cache, reader)
    provenance["eval_file_sha256"] = file_hash(a.eval_file)
    ds = dataset(cfg.data.train_file, cfg, cache, reader, a.limit_train,
                 target_policy=getattr(a, "train_target", "gold"))
    val = dataset(a.eval_file, cfg, cache, reader, a.eval_samples)
    # Check ALL eval IDs, not only the checkpoint-selection prefix.
    all_val = dataset(a.eval_file, cfg, cache, reader)
    if {r["id"] for r in ds.rows} & {r["id"] for r in all_val.rows}:
        raise ValueError("train/eval query-ID overlap")
    del all_val
    target = None
    if a.arm == "direct-state" and a.state_weight > 0:
        if not a.teacher_cache:
            raise ValueError("Direct-State requires --teacher_cache (unless state_weight=0)")
        expected = {k: v for k, v in provenance.items() if k != "eval_file_sha256"}
        target = StateTargets(a.teacher_cache, expected)
        if any(row_key(row) not in target.index for row in ds.rows):
            raise ValueError("teacher cache does not cover the complete selected training set")
        provenance["teacher_manifest_sha256"] = file_hash(Path(a.teacher_cache) / "manifest.json")
    params = [p for p in reader.parameters() if p.requires_grad]
    decoder_params = [p for p in reader.lm.parameters() if p.requires_grad]
    decoder_ids = {id(p) for p in decoder_params}
    added = [p for p in params if id(p) not in decoder_ids]
    groups = [{"params": decoder_params, "lr": a.decoder_lr, "name": "decoder_lora"}]
    if added:
        groups.append({"params": added, "lr": a.workspace_lr, "name": "workspace"})
    optimizer = torch.optim.AdamW(groups, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda_factory(a.steps, a.warmup_ratio))
    best, start = {"score": -float("inf"), "step": None}, 0
    if a.resume:
        ckpt = torch.load(a.resume, map_location="cpu", weights_only=False)
        old_args = dict(ckpt["args"])
        old_args.setdefault("train_target", "gold")
        if training_signature(argparse.Namespace(**old_args)) != training_signature(a):
            raise ValueError("resume arguments differ from the saved trajectory")
        if ckpt["provenance"] != provenance:
            raise ValueError("resume provenance changed")
        reader.restore_state(ckpt["weights"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        restore_rng(ckpt["rng"])
        start, best = ckpt["step"], ckpt["best"]
        del ckpt
    manifest = {"args": vars(a), "provenance": provenance, "environment": environment(),
                "train_rows": len(ds), "validation_rows": len(val),
                "trainable_decoder": sum(p.numel() for p in decoder_params),
                "trainable_workspace": sum(p.numel() for p in added),
                "effective_batch_size": a.batch_size * a.grad_accum,
                "train_target_policy": getattr(a, "train_target", "gold"),
                "p1_prefer_teacher_output": reader.p1_prefer_teacher_output,
                "train_target_counts": {source: sum(r["target_source"] == source for r in ds.rows)
                                        for source in ("gold", "teacher")},
                "train_targets_differ_from_first_gold": sum(r["target"].strip() != r["answers"][0].strip()
                                                             for r in ds.rows),
                "evaluation_note": "dev selection only; independent test remains untouched"}
    atomic_json(out / "manifest.json", manifest)
    val_loader = loader(val, cfg, cache, reader, a.eval_batch_size, a.num_workers)
    def validation(step):
        nonlocal best
        summary, rows = evaluate(reader, val_loader, a.device, a.max_new_tokens)
        score = summary["metrics"]["all"][a.select_metric]
        record = {"step": step, **summary}
        with open(out / "validation.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"[validation] {record}", flush=True)
        if score > best["score"]:
            best = {"score": score, "step": step, "metric": a.select_metric}
            save_checkpoint(out / "checkpoint_best.pt", reader, a, provenance, step, best, optimizer, scheduler)
            atomic_json(out / "best_predictions.json", rows)
            atomic_json(out / "best_checkpoint.json", best)
    if not a.resume:
        validation(0)  # Includes step zero in best selection; continuation can regress.
    sampler = EpochBatches(len(ds), a.batch_size, a.seed, start * a.grad_accum,
                           (a.steps - start) * a.grad_accum)
    batches = iter(loader(ds, cfg, cache, reader, a.batch_size, a.num_workers, sampler))
    reader.train()
    started = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for step in range(start + 1, a.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        totals = {"ce": 0., "state": 0., "loss": 0.}
        for _ in range(a.grad_accum):
            batch = move_to_device(next(batches), a.device)
            packed = reader.pack(batch)
            # Deliberately spans backward: checkpoint recomputation needs the same hooks and Z.
            with reader.activate(packed, capture=target is not None) as captured:
                result = reader.lm(**packed["inputs"], use_cache=False)
                ce = result.loss
                state = ce.new_zeros(())
                if target is not None:
                    state = cosine_state_loss(reader.stack_states(captured), target.gather(batch["raw"], a.device))
                loss = ce + a.state_weight * state
                if not torch.isfinite(loss):
                    raise ValueError("non-finite training loss")
                (loss / a.grad_accum).backward()
                for key, value in (("ce", ce), ("state", state), ("loss", loss)):
                    totals[key] += float(value.detach()) / a.grad_accum
            del result, ce, state, loss, captured, packed
        decoder_grad = torch.stack([p.grad.float().norm() for p in decoder_params if p.grad is not None]).norm()
        added_grad = torch.stack([p.grad.float().norm() for p in added if p.grad is not None]).norm() if added else torch.tensor(0.)
        norm = torch.nn.utils.clip_grad_norm_(params, a.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if step % a.log_every == 0 or step == 1 or step == a.steps:
            record = {"step": step, **totals, "grad_norm": float(norm),
                      "decoder_grad_norm": float(decoder_grad), "workspace_grad_norm": float(added_grad),
                      "lr": scheduler.get_last_lr(), "seconds": time.perf_counter() - started}
            print(record, flush=True)
            with open(out / "train_log.jsonl", "a") as f:
                f.write(json.dumps(record) + "\n")
        if step % a.eval_every == 0 or step == a.steps:
            validation(step)
        if step % a.save_every == 0 or step == a.steps:
            save_checkpoint(out / "checkpoint_last.pt", reader, a, provenance, step, best, optimizer, scheduler)
    if a.steps == 0:
        save_checkpoint(out / "checkpoint_last.pt", reader, a, provenance, 0, best, optimizer, scheduler)
    atomic_json(out / "completion.json", {"completed_step": a.steps, "best": best,
                "seconds_after_initial_eval": time.perf_counter() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None})


def eval_checkpoint(a):
    out = make_output(a.out_dir)
    saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if saved.get("reader_format") != 1:
        raise ValueError("use this entry point with a reader experiment checkpoint")
    original = argparse.Namespace(**saved["args"])
    original.device = a.device
    original.grad_checkpointing = False
    seed_all(original.seed)
    cfg, cache, reader = load_runtime(original)
    current = identity(original, cfg, cache, reader)
    if any(saved["provenance"].get(k) != v for k, v in current.items()):
        raise ValueError("evaluation base model/data identity changed")
    reader.restore_state(saved["weights"])
    ds = dataset(a.eval_file, cfg, cache, reader, a.limit)
    if a.gold_only:
        for row in ds.rows:
            ranks = sorted(row.get("gold_ranks", []))
            ids = row["retrieved_doc_ids"]
            if len(ranks) != 2 or len(set(ranks)) != 2 or min(ranks) < 0 or max(ranks) >= len(ids):
                raise ValueError(f"invalid K=2 gold ranks: {row['id']}")
            row["retrieved_doc_ids"] = [ids[r] for r in ranks]
    batches = loader(ds, cfg, cache, reader, a.eval_batch_size)
    summary, rows = evaluate(reader, batches, a.device, original.max_new_tokens)
    summary.update(checkpoint=str(Path(a.checkpoint).resolve()), step=saved["step"],
                   checkpoint_sha256=file_hash(a.checkpoint), arm=original.arm,
                   eval_file_sha256=file_hash(a.eval_file), gold_only=a.gold_only,
                   conditions={**{k: current[k] for k in (
                       "init_checkpoint_sha256", "cache_manifest_sha256", "tokenizer_sha256",
                       "generator_path", "max_answer_len", "system_prompt")},
                       "max_docs": 2 if a.gold_only else original.max_docs,
                       "max_new_tokens": original.max_new_tokens, "decoding": "greedy",
                       "attn_implementation": original.attn_implementation},
                   environment=environment())
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "predictions.json", rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    args = parser().parse_args()
    {"cache": build_targets, "train": train, "eval": eval_checkpoint}[args.command](args)
