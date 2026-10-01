"""P1 (part 1) of docs/READER_CAUSAL_ORDER_EXECUTION_PLAN.md: K=2 gold, raw/memory mixed.

P0 found a stable raw--memory gap under P1 that lives entirely on bridge
questions.  This script localises it without any annotation.  Only the two gold
paragraphs are shown, and each one is independently rendered as raw text (``R``,
clipped at the compressor's 128 tokens) or as its cached memory (``M``):

    MM  RR  RM  MR      x      orig (retrieval order) / swap

The pure conditions are token-identical to P0's D0 and RG prompts at K=2 (checked
per row), so a K=2 vs K=10 comparison isolates the eight distractors.  Each gold
paragraph also gets a role: the one whose clipped text contains the answer is
the *answer* doc, the other the *bridge* doc; rows where both or neither contain
it are ``ambiguous`` and excluded from role-based contrasts.

    python scripts/eval_gold_mixed.py --label p1 --checkpoint /data02/quro/runs/oscale_P1/checkpoint_last.pt
    python scripts/eval_gold_mixed.py --summarize RUN_DIR [--p0 P0_RUN_DIR]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import apply_arm, get_config
from src import metrics, paths
from src.cache import LatentCache
from src.causal_order import paired_bootstrap
from src.data import load_corpus, read_jsonl
from src.metrics import normalize_answer
from src.model import build_model
from src.prompt import BuiltPrompt, assemble_inputs

PATTERNS = ("MM", "RR", "RM", "MR")
ORDERS = ("orig", "swap")
CONDITIONS = [f"{p}_{o}" for o in ORDERS for p in PATTERNS]
METRICS = ("em", "f1", "substring", "nll")


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--summarize", default=None, help="run directory to summarise")
    p.add_argument("--p0", default=None, help="P0 run directory of the same decoder (K=10 rows)")
    p.add_argument("--preset", default="pisco_hotpot")
    p.add_argument("--label", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--queries", default=None)
    p.add_argument("--corpus", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=None)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def render_mixed(builder, query, segments):
    """``segments`` is ``[(kind, text)]`` in prompt order, kind ``"M"`` or ``"R"``.

    Memory blocks are concatenated with no separator (as in D0) and raw paragraphs
    with a blank line (as in RG); a blank line also separates a raw paragraph from
    a neighbouring memory block.  So MM == D0 and RR == RG token for token.
    """
    parts, prev = [], None
    for kind, text in segments:
        if prev is not None and "R" in (kind, prev):
            parts.append("\n\n")
        parts.append(builder.slot_string(builder.n_mem_tokens) if kind == "M"
                     else builder._clip(text))
        prev = kind
    text = builder._chat(f"Background:\n{''.join(parts)}\n\nQuestion:{query}")
    ids = builder.tok(text, add_special_tokens=False)["input_ids"]
    slots = [i for i, t in enumerate(ids) if t in builder.mem_token_ids]
    want = builder.n_mem_tokens * sum(k == "M" for k, _ in segments)
    if len(slots) != want:
        raise ValueError(f"mixed prompt has {len(slots)} slots, expected {want}")
    return BuiltPrompt(ids, slots)


def contains_answer(builder, text, answers):
    clipped = normalize_answer(builder._clip(text))
    return any(normalize_answer(a) and normalize_answer(a) in clipped for a in answers)


def prepare(rows, model, corpus, cache):
    builder = model.prompt_builders["D0"]
    items = []
    for row in rows:
        ids = [str(x) for x in row["retrieved_doc_ids"]]
        gold = [ids[r] for r in sorted(row["gold_ranks"])]
        if len(gold) != 2 or any(d not in corpus or d not in cache for d in gold):
            raise SystemExit(f"row {row['id']}: gold paragraphs missing from corpus/cache")
        has = [contains_answer(builder, corpus[d], row["answers"]) for d in gold]
        roles = (["answer", "bridge"] if has == [True, False] else
                 ["bridge", "answer"] if has == [False, True] else ["ambiguous"] * 2)
        query = str(row["query"])
        conds = {}
        for order in ORDERS:
            docs = gold if order == "orig" else gold[::-1]
            doc_roles = roles if order == "orig" else roles[::-1]
            for pattern in PATTERNS:
                segs = [(k, corpus[d]) for k, d in zip(pattern, docs)]
                prompt = render_mixed(builder, query, segs)
                conds[f"{pattern}_{order}"] = {
                    "prompt": prompt, "memory_ids": [d for k, d in zip(pattern, docs) if k == "M"],
                    "raw_roles": [r for k, r in zip(pattern, doc_roles) if k == "R"]}
        # The pure conditions must be exactly P0's D0 / RG prompts at K=2.
        d0 = model.prompt_builders["D0"].build(query, 2 * builder.n_mem_tokens)
        rg = model.prompt_builders["RG"].build(query, 0, [corpus[d] for d in gold])
        if conds["MM_orig"]["prompt"].input_ids != d0.input_ids:
            raise RuntimeError(f"row {row['id']}: MM prompt differs from D0")
        if conds["RR_orig"]["prompt"].input_ids != rg.input_ids:
            raise RuntimeError(f"row {row['id']}: RR prompt differs from RG")
        items.append({"row": row, "gold": gold, "has_answer": has,
                      "role_defined": roles[0] != "ambiguous", "conds": conds})
    return items


def soft_batch(cache, conds, device, dtype, hidden):
    width = max(1, max(8 * len(c["memory_ids"]) for c in conds))
    soft = torch.zeros(len(conds), width, hidden, device=device, dtype=dtype)
    mask = torch.zeros(len(conds), width, dtype=torch.bool, device=device)
    for i, c in enumerate(conds):
        if c["memory_ids"]:
            z, _, _ = cache.get_many([c["memory_ids"]], device=device, dtype=dtype)
            z = z.reshape(-1, z.size(-1))
            soft[i, : z.size(0)] = z
            mask[i, : z.size(0)] = True
    return soft, mask


@torch.no_grad()
def run_condition(lm, tok, cache, items, name, targets, batch_size, max_new, device):
    emb = lm.get_input_embeddings()
    dtype, hidden = emb.weight.dtype, emb.weight.size(1)
    out = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        conds = [it["conds"][name] for it in chunk]
        prompts = [c["prompt"] for c in conds]
        soft, mask = soft_batch(cache, conds, device, dtype, hidden)
        tgt = targets[s:s + batch_size]
        packed = assemble_inputs(emb, prompts, soft, mask, target_ids=tgt,
                                 pad_token_id=tok.pad_token_id or 0, pad_side="right")
        labels = packed.pop("labels")
        logits = lm(**packed, use_cache=False).logits[:, :-1].float()
        target = labels[:, 1:]
        tok_nll = F.cross_entropy(logits.transpose(1, 2), target.clamp_min(0), reduction="none")
        keep = (target != -100).float()
        nll = ((tok_nll * keep).sum(1) / keep.sum(1).clamp_min(1)).cpu().tolist()
        del logits, tok_nll
        packed = assemble_inputs(emb, prompts, soft, mask, target_ids=None,
                                 pad_token_id=tok.pad_token_id or 0, pad_side="left")
        ids = lm.generate(inputs_embeds=packed["inputs_embeds"],
                          attention_mask=packed["attention_mask"], max_new_tokens=max_new,
                          do_sample=False, eos_token_id=tok.eos_token_id,
                          pad_token_id=tok.pad_token_id)
        for it, row_ids, n, p in zip(chunk, ids, nll, prompts):
            pred = tok.decode(row_ids.tolist(), skip_special_tokens=True).strip()
            out.append({"pred": pred, **metrics.score(pred, it["row"]["answers"]),
                        "nll": n, "prompt_tokens": len(p.input_ids)})
    return out


def evaluate(a):
    cfg = get_config(a.preset)
    apply_arm(cfg, "P")
    cfg.generator.lora_init = "frozen"
    if a.checkpoint:
        saved = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["config"]
        for key in ("max_budget", "budget_buckets"):
            setattr(cfg.readout, key, saved["readout"][key])
    cfg.revalidate()
    cache = LatentCache(cfg.data.cache_dir)
    cfg.readout.cache_hidden = cache.metadata.hidden_size
    if cache.metadata.doc_max_length != 128 or cache.metadata.latent_size != 8:
        raise SystemExit("expected the 128-token, m=8 PISCO cache")
    queries = a.queries or cfg.data.eval_files["dev"]
    corpus_path = a.corpus or os.path.join(os.path.dirname(queries), "corpus.jsonl")
    corpus = load_corpus([corpus_path])
    stack, model = build_model(cfg, cache_hidden=cache.metadata.hidden_size)
    if a.checkpoint:
        probe = next(n for n, _ in stack.lm.named_parameters() if "lora_A" in n)
        before = dict(stack.lm.named_parameters())[probe].detach().clone()
        model.load(a.checkpoint, strict=False)
        if torch.equal(before, dict(stack.lm.named_parameters())[probe].detach()):
            raise SystemExit("--checkpoint did not change the decoder LoRA")
    lm, tok = stack.lm, stack.tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm.to(device).eval()
    if model.prompt_builders["D0"].n_mem_tokens != cache.metadata.latent_size:
        raise SystemExit("slot block size differs from the cache's m")

    rows = read_jsonl(queries)[: a.limit] if a.limit else read_jsonl(queries)
    items = prepare(rows, model, corpus, cache)
    targets = []
    for it in items:
        ids = tok(" " + it["row"]["answers"][0].strip(), add_special_tokens=False)["input_ids"]
        targets.append(ids[: cfg.data.max_answer_len] + [tok.eos_token_id])
    max_new = a.max_new_tokens or cfg.train.gen_max_new_tokens

    label = a.label or ("ckpt" if a.checkpoint else "pisco")
    out = a.out_dir or os.path.join(paths.RUNS_DIR, "p1_gold_mixed",
                                    f"{label}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out, exist_ok=False)
    results = {}
    for name in CONDITIONS:
        t0 = time.time()
        results[name] = run_condition(lm, tok, cache, items, name, targets,
                                      a.batch_size, max_new, device)
        agg = metrics.aggregate(results[name])
        print(f"[{label}|{name}] EM={agg['em']:.4f} F1={agg['f1']:.4f} "
              f"sub={agg['substring']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    with open(os.path.join(out, "rows.jsonl"), "w", encoding="utf-8") as f:
        for i, it in enumerate(items):
            r = it["row"]
            f.write(json.dumps({
                "id": r["id"], "query": r["query"], "golds": r["answers"],
                "hop_type": r["hop_type"], "is_yes_no": r["is_yes_no"],
                "gold_doc_ids": it["gold"], "has_answer": it["has_answer"],
                "role_defined": it["role_defined"],
                "raw_roles": {n: it["conds"][n]["raw_roles"] for n in CONDITIONS},
                "modes": {n: results[n][i] for n in CONDITIONS}}, ensure_ascii=False) + "\n")
    import transformers
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"]) != 0
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"label": label, "git": git + ("-dirty" if dirty else ""),
                   "preset": a.preset, "queries": queries, "corpus": corpus_path,
                   "cache_dir": cfg.data.cache_dir, "decoder": cfg.generator.name_or_path,
                   "checkpoint": a.checkpoint, "conditions": CONDITIONS,
                   "max_doc_tokens": 128, "max_new_tokens": max_new, "decoding": "greedy",
                   "n": len(items), "transformers": transformers.__version__,
                   "torch": torch.__version__}, f, ensure_ascii=False, indent=2)
    print("[done]", out)


# -- summary ------------------------------------------------------------------

def load_rows(path):
    return [json.loads(l) for l in open(os.path.join(path, "rows.jsonl"), encoding="utf-8")]


def role_cell(r, pattern_roles):
    """Condition name whose raw paragraphs are exactly ``pattern_roles`` (orig order)."""
    for name in CONDITIONS:
        if name.endswith("_orig") and sorted(r["raw_roles"][name]) == sorted(pattern_roles):
            return name
    raise KeyError(pattern_roles)


def summarize(path, p0_path=None):
    rows = load_rows(path)
    label = json.load(open(os.path.join(path, "manifest.json")))["label"]
    p0 = {}
    if p0_path:
        for line in open(os.path.join(p0_path, "rows.jsonl"), encoding="utf-8"):
            x = json.loads(line)
            p0[x["id"]] = x["modes"]
    subsets = {"all": lambda r: True, "bridge": lambda r: r["hop_type"] == "bridge",
               "comparison": lambda r: r["hop_type"] == "comparison",
               "bridge_role": lambda r: r["hop_type"] == "bridge" and r["role_defined"]}

    def value(r, cell, m):
        if cell.startswith("K10_"):
            return p0[r["id"]][cell[4:]][m]
        if cell.startswith("role:"):
            cell = role_cell(r, [x for x in cell[5:].split("+") if x])
        return r["modes"][cell][m]

    report = {"label": label, "n": len(rows), "cells": {}, "contrasts": {}}
    cells = list(CONDITIONS) + (["K10_D0", "K10_RG", "K10_AG"] if p0 else [])
    print(f"{label}: n={len(rows)}")
    print(f"{'cell':<12}" + "".join(f"{s:>22}" for s in subsets))
    for cell in cells:
        line = f"{cell:<12}"
        for s, f in subsets.items():
            keep = [r for r in rows if f(r) and (not cell.startswith("K10") or r["id"] in p0)]
            sub = float(np.mean([value(r, cell, "substring") for r in keep]))
            nll = float(np.mean([value(r, cell, "nll") for r in keep]))
            report["cells"].setdefault(cell, {})[s] = {
                m: float(np.mean([value(r, cell, m) for r in keep])) for m in METRICS}
            report["cells"][cell][s]["n"] = len(keep)
            line += f"   sub {sub:.4f} nll {nll:5.3f}"
        print(line)

    contrasts = [
        ("K=2 gap   RR-MM", "RR_orig", "MM_orig", "all"),
        ("K=2 gap   RR-MM", "RR_orig", "MM_orig", "bridge"),
        ("K=2 gap   RR-MM", "RR_orig", "MM_orig", "comparison"),
        ("order     MM swap-orig", "MM_swap", "MM_orig", "all"),
        ("order     RR swap-orig", "RR_swap", "RR_orig", "all"),
        ("raw bridge doc only", "role:bridge", "role:", "bridge_role"),
        ("raw answer doc only", "role:answer", "role:", "bridge_role"),
        ("both raw vs bridge raw", "role:bridge+answer", "role:bridge", "bridge_role"),
        ("both raw vs answer raw", "role:bridge+answer", "role:answer", "bridge_role"),
    ]
    if p0:
        contrasts += [
            ("K=10 gap  RG-D0", "K10_RG", "K10_D0", "all"),
            ("K=10 gap  RG-D0", "K10_RG", "K10_D0", "bridge"),
            ("distract  MM(K2)-D0(K10)", "MM_orig", "K10_D0", "all"),
            ("distract  MM(K2)-D0(K10)", "MM_orig", "K10_D0", "bridge"),
            ("distract  RR(K2)-RG(K10)", "RR_orig", "K10_RG", "all"),
            ("distract  RR(K2)-RG(K10)", "RR_orig", "K10_RG", "bridge"),
        ]
    print()
    for i, (name, a, b, s) in enumerate(contrasts):
        keep = [r for r in rows if subsets[s](r) and (r["id"] in p0 or "K10" not in a + b)]
        res = {m: paired_bootstrap([value(r, a, m) for r in keep],
                                   [value(r, b, m) for r in keep], seed=i)
               for m in METRICS}
        report["contrasts"][f"{name} [{s}]"] = res
        print(f"{name:<26}{s:<12}n={len(keep):<5}" + "  ".join(
            f"{m} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]"
            for m, v in res.items() if m in ("substring", "em", "nll")))
    with open(os.path.join(path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[summary]", os.path.join(path, "summary.json"))


if __name__ == "__main__":
    args = args_parser()
    if args.summarize:
        summarize(args.summarize, args.p0)
    else:
        evaluate(args)
