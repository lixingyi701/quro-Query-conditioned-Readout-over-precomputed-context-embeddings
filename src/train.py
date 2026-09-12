"""Train/evaluate QuRO v0.0 from cached document latents.

Rows containing teacher_output perform sequence-level knowledge distillation:
the student is trained with next-token CE on the teacher-generated sequence.
Gold answers remain a supported fallback for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import torch
from torch.utils.data import DataLoader

from config import get_config, parse_eval_files
from . import metrics
from .cache import LatentCache
from .data import QuROCollator, RAGCompressionDataset, move_to_device
from .model import build_model


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name):
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available()
                        else ("cpu" if name == "auto" else name))


def cycle(loader):
    while True:
        yield from loader


def lr_lambda_factory(total, warmup_ratio):
    warmup = max(1, int(total * warmup_ratio))
    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return schedule


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="tiny", choices=["tiny", "qwen3emb", "qwen7b"])
    ap.add_argument("--stage", default=None, choices=["stage1"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--num_latents", type=int, default=None, help="prototype offline m")
    ap.add_argument("--num_compressed", type=int, default=None, help="maximum/readout budget B")
    ap.add_argument("--budget_buckets", default=None, help="comma-separated discrete B values")
    ap.add_argument("--adaptive_budget", action="store_true")
    ap.add_argument("--output_query_mode", default=None,
                    choices=["agnostic", "add", "film", "concat", "xattn"])
    ap.add_argument("--without_document_source", action="store_true")
    ap.add_argument("--doc_encoder", default=None,
                    choices=["standalone", "generator_emb", "hf_encoder"])
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--generator", default=None)
    ap.add_argument("--generator_lora", action="store_true")
    ap.add_argument("--projector", choices=["identity", "linear", "mlp"], default=None)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--grad_ckpt", action="store_true")
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--eval_files", default=None)
    ap.add_argument("--cache_dir", default=None,
                    help="QuRO latent cache; omit only for prototype document encoding")
    ap.add_argument("--data_format", default=None, choices=["auto", "rag", "synthetic"])
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--eval_max_samples", type=int, default=None)
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--query_control", action="store_true")
    ap.add_argument("--dump_attn", action="store_true")
    ap.add_argument("--tag", default=None)
    return ap.parse_args()


def apply_overrides(cfg, args):
    mappings = [
        ("stage", cfg.train), ("steps", cfg.train), ("batch_size", cfg.train),
        ("lr", cfg.train), ("device", cfg.train), ("seed", cfg.train),
        ("out_dir", cfg.train), ("resume_from", cfg.train),
        ("grad_accum", cfg.train), ("eval_max_samples", cfg.train),
        ("num_latents", cfg.perceiver), ("num_compressed", cfg.perceiver),
        ("output_query_mode", cfg.perceiver), ("train_file", cfg.data),
        ("max_docs", cfg.data), ("cache_dir", cfg.data),
    ]
    for name, target in mappings:
        value = getattr(args, name)
        if value is not None:
            setattr(target, name, value)
    if args.eval_files:
        cfg.data.eval_files = parse_eval_files(args.eval_files)
    if args.data_format:
        cfg.data.fmt = args.data_format
    if args.budget_buckets:
        cfg.perceiver.budget_buckets = [int(x) for x in args.budget_buckets.split(",")]
    if args.adaptive_budget:
        cfg.perceiver.adaptive_budget = True
    if args.without_document_source:
        cfg.perceiver.add_document_source = False
    if args.doc_encoder:
        cfg.doc_encoder.kind = args.doc_encoder
    if args.encoder:
        cfg.doc_encoder.kind = "hf_encoder"
        cfg.doc_encoder.name_or_path = args.encoder
    if args.generator:
        cfg.generator.name_or_path = args.generator
    if args.generator_lora:
        cfg.generator.lora = True
        cfg.generator.freeze = True
    if args.projector:
        cfg.projector.kind = args.projector
    if args.bf16:
        cfg.train.bf16 = True
        cfg.generator.dtype = "bfloat16"
        cfg.doc_encoder.dtype = "bfloat16"
    if args.grad_ckpt:
        cfg.train.grad_ckpt = True
    cfg.data.prefer_teacher_output = cfg.train.prefer_teacher_output
    cfg.perceiver.__post_init__()
    cfg.doc_encoder.__post_init__()
    cfg.data.__post_init__()
    cfg.train.__post_init__()
    return cfg


def build_loaders(cfg, tokenizer, enc_tokenizer, collator, query_control):
    train = RAGCompressionDataset(
        cfg.data.train_file, tokenizer, cfg.data, enc_tokenizer=enc_tokenizer)
    train_loader = DataLoader(train, batch_size=cfg.train.batch_size, shuffle=True,
                              collate_fn=collator)
    evals = {}
    for name, path in cfg.data.resolved_eval_files().items():
        variants = [(name, 0)]
        if query_control:
            variants.append((name + "/mismatch-q", 1))
        for variant, shift in variants:
            dataset = RAGCompressionDataset(
                path, tokenizer, cfg.data, query_shift=shift,
                enc_tokenizer=enc_tokenizer, limit=cfg.train.eval_max_samples)
            evals[variant] = (dataset, DataLoader(
                dataset, batch_size=cfg.train.batch_size, shuffle=False,
                collate_fn=collator))
    return train, train_loader, evals


@torch.no_grad()
def evaluate(model, loader, device, max_new_tokens, dump_attn_path=None):
    model.eval()
    model.generator.eval()
    rows, attention, total_source, total_budget = [], [], 0, 0
    for batch in loader:
        batch = move_to_device(batch, device)
        predictions = model.generate_answer(batch, max_new_tokens=max_new_tokens)
        for item, prediction in zip(batch["raw"], predictions):
            golds = item.get("answers") or [item["answer"]]
            rows.append({"id": item["id"], "gold": golds[0], "golds": golds,
                         "pred": prediction, **metrics.score(prediction, golds)})
        if "source_token_counts" in batch:
            total_source += int(batch["source_token_counts"].sum())
        if "budget" in batch:
            total_budget += int(batch["budget"].sum())
        else:
            total_budget += len(predictions) * model.cfg.perceiver.num_compressed
        if dump_attn_path:
            latents, doc_mask = model._batch_latents(batch)
            result = model.readout_cached(
                latents, doc_mask, batch["query_ids"], batch["query_mask"],
                budget=batch.get("budget"), return_attn=True)
            attention.append(result["attention"].float().cpu())
    if attention:
        torch.save(attention, dump_attn_path)
    aggregate = metrics.aggregate(rows)
    aggregate["source_tokens"] = total_source
    aggregate["readout_tokens"] = total_budget
    aggregate["xi_eff"] = total_source / total_budget if total_budget else None
    return aggregate, rows


def run_evaluations(model, loaders, device, out_dir, cfg, args, cache):
    result = {
        "tag": args.tag, "version": "0.0.0",
        "output_query_mode": cfg.perceiver.output_query_mode,
        "budget_buckets": cfg.perceiver.budget_buckets,
        "adaptive_budget": cfg.perceiver.adaptive_budget,
        "offline_m": cache.metadata.latent_size if cache else cfg.perceiver.num_latents,
        "cache_compressor": cache.metadata.compressor if cache else "prototype",
        "metrics": {},
    }
    for name, (_, loader) in loaders.items():
        dump = None
        if args.dump_attn and "/" not in name:
            dump = os.path.join(out_dir, "attention_" + name + ".pt")
        aggregate, rows = evaluate(
            model, loader, device, cfg.train.gen_max_new_tokens, dump)
        result["metrics"][name] = aggregate
        filename = "predictions_" + name.replace("/", "_") + ".json"
        with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"[eval] {name}: EM={aggregate['em']:.2%} F1={aggregate['f1']:.3f} "
              f"substring={aggregate['substring']:.2%} xi_eff={aggregate['xi_eff']}")
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    args = build_args()
    cfg = apply_overrides(get_config(args.preset), args)
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)

    cache = LatentCache(cfg.data.cache_dir) if cfg.data.cache_dir else None
    if cache is not None:
        cfg.perceiver.cached_hidden_size = cache.metadata.hidden_size
        cfg.perceiver.num_latents = cache.metadata.latent_size
    elif args.preset != "tiny":
        print("[warn] no --cache_dir: using the prototype encoder, not cache-first QuRO")

    tokenizer, generator, model = build_model(cfg)
    if cache is not None:
        model.set_cache_only()
    generator.to(device)
    model.to(device)
    if cfg.train.grad_ckpt and hasattr(model.doc_encoder, "enable_grad_ckpt"):
        model.doc_encoder.enable_grad_ckpt()
    cfg.to_json(os.path.join(cfg.train.out_dir, "config.json"))

    enc_tokenizer = model.enc_tok if model.enc_tok is not tokenizer else None
    collator = QuROCollator(
        model.pad_id, model.enc_pad_id, cache=cache, max_docs=cfg.data.max_docs,
        require_budget_labels=cfg.perceiver.adaptive_budget and not args.eval_only)
    train_set, train_loader, eval_loaders = build_loaders(
        cfg, tokenizer, enc_tokenizer, collator, args.query_control)
    print(f"[info] QuRO v0.0 device={device} train={len(train_set)} "
          f"cache={'yes' if cache else 'prototype'} trainable={model.num_trainable()/1e6:.2f}M")

    if args.eval_only:
        if cfg.train.resume_from:
            model.load(cfg.train.resume_from)
        run_evaluations(model, eval_loaders, device, cfg.train.out_dir, cfg, args, cache)
        return

    params = [p for p in model.parameters() if p.requires_grad]
    params += [p for p in generator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg.train.steps, cfg.train.warmup_ratio))
    start_step = 0
    if cfg.train.resume_from:
        _, _, saved_step = model.load(
            cfg.train.resume_from, optimizer=optimizer, scheduler=scheduler)
        start_step = int(saved_step or 0)

    iterator = cycle(train_loader)
    model.train()
    if cfg.generator.lora:
        generator.train()
    log_path = os.path.join(cfg.train.out_dir, "train_log.jsonl")
    started = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        for step in range(start_step, cfg.train.steps):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = budget_sum = 0.0
            for _ in range(max(1, cfg.train.grad_accum)):
                batch = move_to_device(next(iterator), device)
                output = model(batch, beta=cfg.train.beta_qa)
                (output["loss"] / cfg.train.grad_accum).backward()
                loss_sum += float(output["loss"]) / cfg.train.grad_accum
                budget_sum += float(output["mean_budget"]) / cfg.train.grad_accum
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            optimizer.step()
            scheduler.step()
            if step % cfg.train.log_every == 0 or step + 1 == cfg.train.steps:
                record = {"step": step + 1, "loss": loss_sum,
                          "mean_budget": budget_sum, "grad_norm": float(grad_norm),
                          "lr": scheduler.get_last_lr()[0],
                          "seconds": round(time.time() - started, 1)}
                print(record)
                log.write(json.dumps(record) + "\n")
                log.flush()

    checkpoint = os.path.join(cfg.train.out_dir, "checkpoint_last.pt")
    model.save(checkpoint, optimizer=optimizer, scheduler=scheduler, step=cfg.train.steps)
    run_evaluations(model, eval_loaders, device, cfg.train.out_dir, cfg, args, cache)


if __name__ == "__main__":
    main()
