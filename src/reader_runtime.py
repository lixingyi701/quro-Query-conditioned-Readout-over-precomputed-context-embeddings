"""Shared provenance, loading, batches and evaluation for the three reader arms."""
from __future__ import annotations

from dataclasses import asdict, fields
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from config import get_config
from .cache import LatentCache
from .data import QuROCollator, QuRODataset, move_to_device
from .model import build_model
from . import metrics
from .reader_experiment import ReaderExperiment, ReaderSpec, file_hash, json_hash, tokenizer_hash


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def environment():
    import transformers
    import peft
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return {"git": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "peft": peft.__version__, "cuda": torch.version.cuda}


def load_runtime(args, teacher=False):
    """Warm-start P1 weights only; never inherit its optimizer/scheduler."""
    saved = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
    if saved.get("reader_format"):
        raise ValueError("--init_checkpoint must be the original P1, use --resume for continuation")
    cfg = get_config(args.preset)
    for section in ("readout", "query_encoder", "generator", "decoder", "data", "train"):
        target = getattr(cfg, section)
        known = {f.name for f in fields(target)}
        for key, value in saved["config"][section].items():
            if key in known:
                setattr(target, key, value)
    if cfg.generator.kind != "pisco" or cfg.readout.kind != "pisco_direct":
        raise ValueError("this protocol requires a PISCO direct-memory P1 checkpoint")
    if cfg.decoder.input_mode != "D0" or cfg.readout.output_scale != 1.0 or cfg.readout.output_scale_learnable:
        raise ValueError("P1 must use D0 and fixed output_scale=1")
    if cfg.readout.adaptive_budget:
        raise ValueError("adaptive budgets are outside this protocol")
    cfg.generator.lora_init = "pisco"
    cfg.generator.device = args.device
    if args.generator_path:
        cfg.generator.name_or_path = args.generator_path
    cfg.generator.attn_implementation = args.attn_implementation
    cfg.data.cache_dir = args.cache_dir or cfg.data.cache_dir
    cfg.data.train_file = args.train_file or cfg.data.train_file
    cfg.data.max_docs = args.max_docs
    cfg.data.prefer_teacher_output = False
    cfg.decoder.query_text_dropout = 0.0
    cfg.train.grad_ckpt = False  # enabled explicitly and identically below
    cfg.revalidate()
    cache = LatentCache(cfg.data.cache_dir)
    if cache.metadata.doc_max_length != 128 or cache.metadata.latent_size != 8:
        raise ValueError("expected PISCO cache: 128 source tokens, 8 slots/document")
    stack, base = build_model(cfg, cache_hidden=cache.metadata.hidden_size)
    current = dict(base.lm.named_parameters())
    payload = saved.get("generator_trainable", {})
    trainable = {n for n, p in current.items() if p.requires_grad}
    if not trainable or set(payload) != trainable or any("lora_" not in n for n in trainable):
        raise ValueError("P1 decoder payload must exactly cover the active trainable LoRA")
    base.load(args.init_checkpoint)
    for name, tensor in payload.items():
        if not torch.equal(current[name].detach().cpu(), tensor.to(current[name].dtype)):
            raise ValueError(f"P1 LoRA restoration failed: {name}")
    spec = ReaderSpec(args.arm, tuple(int(x) for x in args.layers.split(",")),
                      args.state_weight, args.workspace_tokens, args.cross_dim,
                      args.cross_heads, args.gate_init)
    reader = ReaderExperiment(base, spec).to(args.device)
    if teacher:
        reader.requires_grad_(False)
        reader.eval()
    elif args.grad_checkpointing:
        reader.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # The online student holds only the decoder. Free the unused offline stack.
    del stack, base, saved
    return cfg, cache, reader


def identity(args, cfg, cache, reader):
    builder = reader.builders["D0"]
    return {"init_checkpoint_sha256": file_hash(args.init_checkpoint),
            "train_file_sha256": file_hash(cfg.data.train_file),
            "cache_manifest_sha256": file_hash(Path(cfg.data.cache_dir) / "manifest.json"),
            "cache_path": str(Path(cfg.data.cache_dir).resolve()),
            "cache_metadata": asdict(cache.metadata),
            "generator_path": str(Path(cfg.generator.name_or_path).resolve()),
            "tokenizer_sha256": tokenizer_hash(reader.tok),
            "system_prompt": builder.system_prompt, "max_doc_tokens": builder.max_doc_tokens,
            "max_prompt_tokens": builder.max_prompt_tokens, "max_docs": cfg.data.max_docs,
            "max_answer_len": cfg.data.max_answer_len, "target_source": "gold",
            "layers": list(reader.spec.layers), "hidden": reader.hidden}


def dataset(path, cfg, cache, reader, limit=None, corpus=None):
    ds = QuRODataset(path, reader.tok, cfg.data, limit=limit, corpus=corpus)
    ids = [r["id"] for r in ds.rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("dataset must be nonempty with unique query IDs")
    for row in ds.rows:
        if not row["answers"] or row["target_source"] != "gold":
            raise ValueError("every row requires a gold answer")
        for doc in row["retrieved_doc_ids"]:
            if doc not in cache or (corpus is not None and doc not in corpus):
                raise ValueError(f"missing document {doc} for row {row['id']}")
    return ds


def collator(cfg, cache, reader):
    return QuROCollator(cache, reader.pad_id, max_docs=cfg.data.max_docs)


class EpochBatches(Sampler):
    """Deterministic, resumable microbatch order independent of model RNG draws."""
    def __init__(self, size, batch_size, seed, start=0, count=1):
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.start, self.count = start, count
        if size < 1 or batch_size < 1:
            raise ValueError("empty dataset or invalid batch size")

    def __len__(self):
        return self.count

    def __iter__(self):
        per_epoch = math.ceil(self.size / self.batch_size)
        cached_epoch, indices = None, None
        for batch in range(self.start, self.start + self.count):
            epoch, offset = divmod(batch, per_epoch)
            if cached_epoch != epoch:
                generator = torch.Generator().manual_seed(self.seed + epoch)
                indices = torch.randperm(self.size, generator=generator).tolist()
                cached_epoch = epoch
            yield indices[offset * self.batch_size:(offset + 1) * self.batch_size]


def loader(ds, cfg, cache, reader, batch_size, workers=0, sampler=None):
    # DataLoader worker seed generation must not perturb LoRA dropout RNG.
    options = {"num_workers": workers, "collate_fn": collator(cfg, cache, reader),
               "generator": torch.Generator().manual_seed(991)}
    if sampler is None:
        options.update(batch_size=batch_size, shuffle=False)
    else:
        options["batch_sampler"] = sampler
    return DataLoader(ds, **options)


@torch.no_grad()
def evaluate(reader, batches, device, max_new_tokens):
    previous = reader.training
    reader.eval()
    records = []
    started = time.perf_counter()
    try:
        for batch in batches:
            batch = move_to_device(batch, device)
            packed = reader.pack(batch)
            with reader.activate(packed):
                out = reader.lm(**packed["inputs"], use_cache=False)
            labels = packed["inputs"]["labels"][:, 1:]
            keep = labels != -100
            losses = torch.nn.functional.cross_entropy(
                out.logits[:, :-1].float().transpose(1, 2), labels.clamp_min(0), reduction="none")
            nlls = (losses * keep).sum(1) / keep.sum(1).clamp_min(1)
            if not torch.isfinite(nlls).all():
                raise ValueError("non-finite evaluation NLL")
            nlls = nlls.cpu().tolist()
            del out, losses
            predictions = reader.generate(batch, max_new_tokens)
            for row, pred, nll, n_prompt in zip(batch["raw"], predictions, nlls, packed["prompt_lengths"]):
                records.append({"id": row["id"], "golds": row["answers"], "pred": pred,
                                "hop_type": row.get("hop_type"), "nll": nll,
                                "prompt_tokens": n_prompt, **metrics.score(pred, row["answers"])})
    finally:
        reader.train(previous)
    summaries = {}
    for split in ("all", "bridge", "comparison"):
        selected = records if split == "all" else [r for r in records if r["hop_type"] == split]
        if selected:
            summaries[split] = {**metrics.aggregate(selected),
                                "nll": float(np.mean([r["nll"] for r in selected])),
                                "mean_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in selected]))}
    return {"metrics": summaries, "eval_seconds": time.perf_counter() - started,
            "timing_note": "generation plus teacher-forced NLL; not isolated answer latency"}, records


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, reader, args, provenance, step, best, optimizer, scheduler):
    path = Path(path)
    payload = {"reader_format": 1, "spec": asdict(reader.spec), "args": vars(args),
               "provenance": provenance, "weights": reader.checkpoint_state(),
               "step": step, "best": best, "rng": rng_state(),
               "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def training_signature(args):
    # All choices affecting the trajectory must match on resume; output paths may move.
    excluded = {"command", "out_dir", "resume", "num_workers", "device"}
    return json_hash({k: v for k, v in vars(args).items() if k not in excluded})
