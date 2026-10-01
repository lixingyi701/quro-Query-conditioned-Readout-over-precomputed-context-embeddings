"""P0 of docs/READER_CAUSAL_ORDER_EXECUTION_PLAN.md: is there a stable raw--memory gap?

One decoder, one set of weights, one batch of HotpotQA rows; only what fills the
evidence slot changes:

``D0``  PISCO's compressed memory (the cached latents, all ``K * m`` of them)
``RG``  the same retrieved documents as raw text, clipped at the 128 tokens the
        compressor saw, under the same ``Background:/Question:`` scaffolding
``AG``  no evidence at all -- the parametric-knowledge floor

Run it once per decoder (released PISCO adapter, the P1 checkpoint, optionally the
bare Mistral with ``--disable_adapter``).  Every row records generation scores,
the output length and the teacher-forced answer NLL, so ``--summarize`` can do
paired, per-row comparisons between any two (run, mode) cells.  Under a memory-
adapted decoder RG is a same-weights representational control, not an upper bound.

    python scripts/eval_raw_vs_memory.py --label pisco --modes D0,RG,AG
    python scripts/eval_raw_vs_memory.py --label p1 --checkpoint /data02/quro/runs/oscale_P1/checkpoint_last.pt
    python scripts/eval_raw_vs_memory.py --summarize RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import apply_arm, get_config
from src import metrics, paths
from src.cache import LatentCache
from src.causal_order import paired_bootstrap
from src.data import QuROCollator, QuRODataset, load_corpus, move_to_device
from src.model import build_model
from src.prompt import assemble_inputs

METRICS = ("em", "f1", "substring")


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--summarize", nargs="+", default=None,
                   help="run directories to compare instead of evaluating")
    p.add_argument("--preset", default="pisco_hotpot")
    p.add_argument("--label", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--disable_adapter", action="store_true",
                   help="bare Mistral-7B-Instruct-v0.2; only RG/AG are meaningful")
    p.add_argument("--modes", default="D0,RG,AG")
    p.add_argument("--queries", default=None)
    p.add_argument("--corpus", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def git_sha():
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"]) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


@torch.no_grad()
def per_row_nll(model, batch):
    """Mean teacher-forced NLL of the gold answer, one value per row.

    ``qa_loss`` averages over the whole batch, which would weight long answers
    more and makes paired per-row comparisons impossible.
    """
    result = model.readout_cached(batch)
    prompts = model.build_prompts(batch, result["soft_token_mask"], training=False)
    packed = assemble_inputs(
        model.lm.get_input_embeddings(), prompts,
        result["soft_tokens"], result["soft_token_mask"],
        target_ids=batch["target_ids"], pad_token_id=model.pad_id, pad_side="right")
    labels = packed.pop("labels")
    logits = model.lm(**packed, use_cache=False).logits[:, :-1].float()
    target = labels[:, 1:]
    token_nll = F.cross_entropy(logits.transpose(1, 2), target.clamp_min(0),
                                reduction="none")
    keep = (target != -100).float()
    nll = (token_nll * keep).sum(1) / keep.sum(1).clamp_min(1)
    if not torch.isfinite(nll).all():
        raise ValueError("non-finite answer NLL; inspect the batch")
    return nll.cpu().tolist(), [len(p.input_ids) for p in prompts]


def evaluate(a):
    cfg = get_config(a.preset)
    apply_arm(cfg, "P")
    # "frozen" loads the released PISCO adapter without making it trainable; a
    # checkpoint then overwrites those LoRA weights with the trained ones.
    cfg.generator.lora_init = "frozen"
    if a.checkpoint:
        # Buffer shapes (budget buckets) must match the run that wrote it.
        saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["config"]
        for key in ("max_budget", "budget_buckets"):
            setattr(cfg.readout, key, saved["readout"][key])
    cfg.revalidate()
    cache = LatentCache(cfg.data.cache_dir)
    cfg.readout.cache_hidden = cache.metadata.hidden_size
    if cache.metadata.doc_max_length != 128:
        raise SystemExit("RG clips documents at 128 tokens; the cache was built differently")
    queries = a.queries or cfg.data.eval_files["dev"]
    corpus_path = a.corpus or os.path.join(os.path.dirname(queries), "corpus.jsonl")
    corpus = load_corpus([corpus_path])

    stack, model = build_model(cfg, cache_hidden=cache.metadata.hidden_size)
    if a.checkpoint:
        probe = next(n for n, _ in stack.lm.named_parameters() if "lora_A" in n)
        before = dict(stack.lm.named_parameters())[probe].detach().clone()
        model.load(a.checkpoint, strict=False)
        after = dict(stack.lm.named_parameters())[probe].detach()
        if torch.equal(before, after):
            raise SystemExit(f"--checkpoint did not change the decoder LoRA ({probe})")
    if a.disable_adapter:
        stack.lm.disable_adapters()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    stack.lm.to(device)
    model.eval()

    dataset = QuRODataset(queries, stack.tokenizer, cfg.data,
                          query_tokenizer=stack.query_tokenizer, limit=a.limit,
                          corpus=corpus)
    # The dataset silently drops documents missing from the corpus; RG would then
    # be judged on less evidence than D0.  Refuse instead.
    k = cfg.data.max_docs
    for row in dataset.rows:
        ids = row["retrieved_doc_ids"][:k] if k else row["retrieved_doc_ids"]
        missing = [d for d in ids if d not in corpus or d not in cache]
        if missing:
            raise SystemExit(f"row {row['id']} lacks {len(missing)} documents in corpus/cache")
    collator = QuROCollator(cache, pad_id=model.pad_id,
                            query_pad_id=getattr(stack.query_tokenizer, "pad_token_id", model.pad_id),
                            max_docs=k)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, collate_fn=collator)
    max_new = a.max_new_tokens or cfg.train.gen_max_new_tokens
    modes = a.modes.split(",")

    label = a.label or ("bare" if a.disable_adapter else "ckpt" if a.checkpoint else "pisco")
    out = a.out_dir or os.path.join(paths.RUNS_DIR, "p0_raw_memory",
                                    f"{label}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out, exist_ok=False)
    rows = {}
    for mode in modes:
        model.decoder_input_mode = mode
        t0 = time.time()
        for batch in loader:
            batch = move_to_device(batch, device)
            if mode == "RG":
                # 14 dev rows natively retrieve fewer than K documents; D0 sees the
                # same reduced set, so only a mismatch with the row itself is an error.
                for texts, item in zip(batch["document_texts"], batch["raw"]):
                    want = len(item["retrieved_doc_ids"][:k] if k else item["retrieved_doc_ids"])
                    if len(texts) != want:
                        raise RuntimeError(f"RG row {item['id']} has {len(texts)} documents, expected {want}")
            nll, prompt_len = per_row_nll(model, batch)
            preds = model.generate_answer(batch, max_new_tokens=max_new)
            for i, item in enumerate(batch["raw"]):
                golds = item.get("answers") or [item["answer"]]
                r = rows.setdefault(item["id"], {
                    "id": item["id"], "query": item["query"], "golds": golds,
                    "hop_type": item.get("hop_type"), "is_yes_no": item.get("is_yes_no"),
                    "modes": {}})
                r["modes"][mode] = {
                    "pred": preds[i], **metrics.score(preds[i], golds),
                    "nll": nll[i], "prompt_tokens": prompt_len[i],
                    "output_tokens": len(stack.tokenizer(preds[i], add_special_tokens=False)["input_ids"]),
                }
        agg = metrics.aggregate([r["modes"][mode] for r in rows.values()])
        print(f"[{label}|{mode}] EM={agg['em']:.4f} F1={agg['f1']:.4f} "
              f"sub={agg['substring']:.4f} n={agg['n']} ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(out, "rows.jsonl"), "w", encoding="utf-8") as f:
        for r in rows.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import transformers
    manifest = {
        "label": label, "git": git_sha(), "preset": a.preset, "queries": queries,
        "corpus": corpus_path, "cache_dir": cfg.data.cache_dir,
        "cache": cache.metadata.__dict__, "decoder": cfg.generator.name_or_path,
        "checkpoint": a.checkpoint, "disable_adapter": a.disable_adapter,
        "modes": modes, "max_docs": k, "max_doc_tokens": 128, "max_new_tokens": max_new,
        "decoding": "greedy", "batch_size": a.batch_size, "limit": a.limit,
        "n": len(rows), "seed": a.seed, "torch": torch.__version__,
        "transformers": transformers.__version__, "python": platform.python_version(),
    }
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[done]", out)


def load_run(path):
    manifest = json.load(open(os.path.join(path, "manifest.json")))
    rows = {}
    for line in open(os.path.join(path, "rows.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        rows[r["id"]] = r
    return manifest["label"], rows


def cell_summary(rows, mode, subset=None):
    xs = [r["modes"][mode] for r in rows if subset is None or r["hop_type"] == subset]
    out = {m: float(np.mean([x[m] for x in xs])) for m in METRICS}
    out["nll"] = float(np.mean([x["nll"] for x in xs]))
    out["output_tokens"] = float(np.mean([x["output_tokens"] for x in xs]))
    out["prompt_tokens"] = float(np.mean([x["prompt_tokens"] for x in xs]))
    out["n"] = len(xs)
    return out


def paired(rows, a, b, metric, subset=None, seed=0):
    """Paired bootstrap of cell ``a`` minus cell ``b``; cells are (label, mode)."""
    keep = [r for r in rows if subset is None or r[a[0]]["hop_type"] == subset]
    x = [r[a[0]]["modes"][a[1]][metric] for r in keep]
    y = [r[b[0]]["modes"][b[1]][metric] for r in keep]
    return paired_bootstrap(x, y, seed=seed)


def summarize(paths_):
    runs = dict(load_run(p) for p in paths_)
    ids = set.intersection(*(set(r) for r in runs.values()))
    if any(len(r) != len(ids) for r in runs.values()):
        print(f"[warn] runs cover different rows; comparing the {len(ids)} shared ones")
    joined = [{label: runs[label][i] for label in runs} for i in sorted(ids)]
    cells = {}
    for label, rows in runs.items():
        shared = [rows[i] for i in sorted(ids)]
        for mode in next(iter(shared))["modes"]:
            cells[f"{label}|{mode}"] = {s or "all": cell_summary(shared, mode, s)
                                        for s in (None, "bridge", "comparison")}
    contrasts = {}
    for label in runs:
        modes = next(iter(runs[label].values()))["modes"]
        pairs = [("RG", "D0"), ("D0", "AG"), ("RG", "AG")]
        for x, y in pairs:
            if x in modes and y in modes:
                key = f"{label}: {x} - {y}"
                contrasts[key] = {
                    s or "all": {m: paired(joined, (label, x), (label, y), m, s)
                                 for m in METRICS + ("nll",)}
                    for s in (None, "bridge", "comparison")}
    report = {"n_shared": len(ids), "cells": cells, "contrasts": contrasts}
    print(f"{'cell':<16}{'EM':>8}{'F1':>8}{'sub':>8}{'NLL':>8}{'out_tok':>9}{'prompt':>8}")
    for key, c in cells.items():
        x = c["all"]
        print(f"{key:<16}{x['em']:>8.4f}{x['f1']:>8.4f}{x['substring']:>8.4f}"
              f"{x['nll']:>8.3f}{x['output_tokens']:>9.1f}{x['prompt_tokens']:>8.0f}")
    print()
    for key, c in contrasts.items():
        for s in ("all", "bridge", "comparison"):
            parts = "  ".join(f"{m} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]"
                              for m, v in c[s].items())
            print(f"{key:<22}{s:<11}{parts}")
    out = os.path.join(paths.RUNS_DIR, "p0_raw_memory", "summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[summary]", out)


if __name__ == "__main__":
    args = args_parser()
    if args.summarize:
        summarize(args.summarize)
    else:
        evaluate(args)
