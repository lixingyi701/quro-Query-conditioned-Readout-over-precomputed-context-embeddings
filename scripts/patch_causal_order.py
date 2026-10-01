"""Scoped P3 of docs/READER_CAUSAL_ORDER_EXECUTION_PLAN.md: does D2's Q->M state matter?

P2 found that in the question-first layout (D2) the question moves memory-
position states by ~1e-4 (1-cos) / ~1.3% (relative L2).  That says the path
exists, not that the answer uses it.  This script intervenes on the path in a
fixed decoder and measures the gold answer's teacher-forced NLL and greedy QA:

``none``       plain D2
``id_all``     memory region overwritten by its own states at every block
               (sanity: must change nothing)
``block_QM``   the memory region (slots and the <SEP>s between them) may not
               attend to the question or to the template between question and
               memory -- every position that carries question information.
               Memory states become question-independent (checked).
``xq_all``     memory region overwritten at every block by the states it has
               under a *different* real question with the same documents
               (exact-position partner) -- i.e. memory conditioned on the wrong Q
``xq@l``       the same at a single block output ``l`` (layer choice on the dev
               half, confirmation on the holdout half)
``xdoc_all``   memory region from the partner's *documents* under the same
``xdoc@l``     question: positive control, must hurt, else patching is inert

``block_QM`` and ``xq_all`` involve no layer selection and are the primary tests.

    python scripts/patch_causal_order.py --label pisco --pairs 200
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import diagnose_causal_order as dco  # noqa: E402
from config import apply_arm, get_config  # noqa: E402
from src import metrics, paths  # noqa: E402
from src.cache import LatentCache  # noqa: E402
from src.causal_order import decoder_layers, paired_bootstrap  # noqa: E402
from src.data import read_jsonl  # noqa: E402
from src.prompt import assemble_inputs  # noqa: E402
from src.model import build_model  # noqa: E402

SINGLE_LAYERS = (4, 12, 20)


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--summarize", default=None)
    p.add_argument("--preset", default="pisco_hotpot")
    p.add_argument("--label", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--pairs", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=32)
    p.add_argument("--out_dir", default=None)
    return p.parse_args()


def interventions():
    names = ["none", "id_all", "block_QM", "xq_all", "xdoc_all"]
    names += [f"xq@{l}" for l in SINGLE_LAYERS] + [f"xdoc@{l}" for l in SINGLE_LAYERS]
    return names


# -- mechanics ----------------------------------------------------------------

@contextmanager
def patch_memory(layers, region, donor, which):
    """Overwrite ``region`` positions of block outputs with ``donor[l]``.

    Only applied to forwards that contain the prompt (prefill or teacher-forced),
    never to single-token decode steps, whose position 0 is a new token.
    """
    handles = []
    index = torch.as_tensor(region)

    def hook(l):
        def fn(_m, _a, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if hidden.size(1) <= region[-1]:
                return None
            hidden[0, index.to(hidden.device)] = donor[l].to(hidden.dtype)
            return None
        return fn

    for l in which:
        handles.append(layers[l].register_forward_hook(hook(l)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def nothing():
    yield


@torch.no_grad()
def capture_region(lm, layers, embeds, region, target=None):
    """Region states at every block output.

    Captured in the same forward shape the patched run will use -- prompt plus
    answer for the NLL, prompt with a KV cache for generation -- so that the
    identity patch is bit-exact rather than bf16-kernel-noisy.
    """
    use_cache = target is None
    if target is not None:
        embeds = torch.cat([embeds, lm.get_input_embeddings()(target[None])], dim=1)
    store = {}
    index = torch.as_tensor(region)
    handles = [layer.register_forward_hook(
        lambda _m, _a, o, l=l: store.__setitem__(
            l, (o[0] if isinstance(o, (tuple, list)) else o)[0, index.to(embeds.device)].clone()))
        for l, layer in enumerate(layers)]
    try:
        lm(inputs_embeds=embeds, use_cache=use_cache)
    finally:
        for h in handles:
            h.remove()
    return [store[l] for l in range(len(layers))]


def block_mask(length, region, blocked_cols, dtype, device):
    """Additive 4D causal mask; ``region`` rows additionally cannot see ``blocked_cols``."""
    allowed = torch.ones(length, length, dtype=torch.bool, device=device).tril()
    if blocked_cols:
        rows = torch.as_tensor(region, device=device)[:, None]
        cols = torch.as_tensor(blocked_cols, device=device)[None, :]
        allowed[rows, cols] = False
    mask = torch.zeros(length, length, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]


@torch.no_grad()
def answer_nll(lm, embeds, target, mask=None):
    full = torch.cat([embeds, lm.get_input_embeddings()(target[None])], dim=1)
    if mask is not None:
        mask = block_mask(full.size(1), *mask, dtype=full.dtype, device=full.device)
    logits = lm(inputs_embeds=full, attention_mask=mask, use_cache=False).logits[0].float()
    n = target.numel()
    nll = F.cross_entropy(logits[-n - 1:-1], target)
    if not torch.isfinite(nll):
        raise ValueError("non-finite answer NLL")
    return float(nll)


@torch.no_grad()
def greedy(lm, tok, embeds, max_new, mask=None):
    if mask is not None:
        mask = block_mask(embeds.size(1), *mask, dtype=embeds.dtype, device=embeds.device)
    out = lm(inputs_embeds=embeds, attention_mask=mask, use_cache=True)
    past, nxt, ids = out.past_key_values, out.logits[0, -1].argmax(), []
    emb = lm.get_input_embeddings()
    for _ in range(max_new):
        if int(nxt) == tok.eos_token_id:
            break
        ids.append(int(nxt))
        out = lm(inputs_embeds=emb(nxt.view(1, 1)), past_key_values=past, use_cache=True)
        past, nxt = out.past_key_values, out.logits[0, -1].argmax()
    return tok.decode(ids, skip_special_tokens=True).strip()


# -- experiment ---------------------------------------------------------------

def prompt_embeds(lm, tok, builder, cache, ids, query, device):
    prompt, qpos, packed = dco.pack(lm, tok, builder, cache, ids, query, [], device)
    return prompt, qpos, packed["inputs_embeds"][:, : len(prompt.input_ids)]


def evaluate(a):
    torch.manual_seed(a.seed)
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
    stack, model = build_model(cfg, cache_hidden=cache.metadata.hidden_size)
    if a.checkpoint:
        model.load(a.checkpoint, strict=False)
    lm, tok = stack.lm, stack.tokenizer
    device = torch.device("cuda")
    lm.to(device).eval()
    layers = list(decoder_layers(lm))
    builder = model.prompt_builders["D2"]
    rows = read_jsonl(cfg.data.eval_files["dev"])
    pairs = dco.select_pairs(rows, model, tok, cache, a.pairs, cfg.data.max_docs, a.seed)

    label = a.label or ("ckpt" if a.checkpoint else "pisco")
    out = a.out_dir or os.path.join(paths.RUNS_DIR, "p3_patch", f"{label}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out, exist_ok=False)
    names = interventions()
    records = []
    t0 = time.time()
    for pi, (ra, rb) in enumerate(pairs):
        for receiver, partner in ((ra, rb), (rb, ra)):
            ids_r, ids_p = dco.doc_ids(receiver, cfg.data.max_docs), dco.doc_ids(partner, cfg.data.max_docs)
            q_r, q_p = str(receiver["query"]), str(partner["query"])
            golds = receiver["answers"]
            target = tok(" " + golds[0].strip(), add_special_tokens=False)["input_ids"]
            target = torch.tensor(target[: cfg.data.max_answer_len] + [tok.eos_token_id], device=device)

            p_rr, qpos, e_rr = prompt_embeds(lm, tok, builder, cache, ids_r, q_r, device)
            p_rp, qpos_p, e_rp = prompt_embeds(lm, tok, builder, cache, ids_r, q_p, device)
            p_pr, qpos_d, e_pr = prompt_embeds(lm, tok, builder, cache, ids_p, q_r, device)
            if not (p_rr.slot_positions == p_rp.slot_positions == p_pr.slot_positions
                    and qpos == qpos_p == qpos_d):
                raise RuntimeError("position control failed")
            slots = p_rr.slot_positions
            region = list(range(slots[0], slots[-1] + 1))
            blocked = list(range(qpos[0], slots[0]))   # question + template after it
            # Donors under the receiver's answer suffix (for the NLL) and as a
            # cached prefill (for generation); the suffix cannot reach the region.
            donors = {use: [capture_region(lm, layers, e, region, target if use == "nll" else None)
                            for e in (e_rr, e_rp, e_pr)] for use in ("nll", "gen")}

            if len(records) < 8:
                # Blocking must make memory question-independent, exactly as far
                # as the numerics allow; otherwise the mask leaks and nothing
                # downstream is interpretable.
                m = (region, blocked)
                a_states = blocked_capture(lm, layers, e_rr, region, m)
                b_states = blocked_capture(lm, layers, e_rp, region, m)
                diff = max(float((x - y).abs().max()) for x, y in zip(a_states, b_states))
                scale = float(a_states[-1].abs().max())
                if diff > 1e-3 * scale:
                    raise RuntimeError(f"block_QM leaks question information: {diff} vs scale {scale}")
                plain = answer_nll(lm, e_rr, target)
                masked = answer_nll(lm, e_rr, target, (region, []))
                if abs(plain - masked) > 2e-2:
                    raise RuntimeError(f"an all-allowed 4D mask changes the NLL: {plain} vs {masked}")

            rec = {"pair": pi, "id": str(receiver["id"]), "partner_id": str(partner["id"]),
                   "query": q_r, "partner_query": q_p, "golds": golds,
                   "partner_golds": partner["answers"], "hop_type": receiver.get("hop_type"),
                   "n_region": len(region), "n_blocked": len(blocked), "modes": {}}
            for name in names:
                mask = (region, blocked) if name == "block_QM" else None
                with make_ctx(name, layers, region, *donors["nll"]):
                    nll = answer_nll(lm, e_rr, target, mask)
                with make_ctx(name, layers, region, *donors["gen"]):
                    pred = greedy(lm, tok, e_rr, a.max_new_tokens, mask)
                rec["modes"][name] = {"nll": nll, "pred": pred, **metrics.score(pred, golds),
                                      "partner_substring": metrics.score(pred, partner["answers"])["substring"]}
            records.append(rec)
        if (pi + 1) % 10 == 0:
            print(f"[{pi + 1}/{len(pairs)}] {time.time() - t0:.0f}s", flush=True)

    with open(os.path.join(out, "rows.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    import transformers
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.call(["git", "diff", "--quiet", "HEAD"]) != 0
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"label": label, "git": git + ("-dirty" if dirty else ""), "preset": a.preset,
                   "checkpoint": a.checkpoint, "decoder": cfg.generator.name_or_path,
                   "cache_dir": cfg.data.cache_dir, "pairs": len(pairs), "receivers": len(records),
                   "seed": a.seed, "single_layers": SINGLE_LAYERS, "interventions": names,
                   "dev_half": "pair < pairs/2", "max_new_tokens": a.max_new_tokens,
                   "patched_region": "first slot .. last slot (slots and <SEP>s)",
                   "blocked_columns": "first question token .. first slot - 1",
                   "transformers": transformers.__version__, "torch": torch.__version__},
                  f, ensure_ascii=False, indent=2)
    print("[done]", out)


def make_ctx(name, layers, region, self_states, xq_states, xdoc_states):
    """A fresh context per forward: generator context managers are single-use."""
    every = range(len(layers))
    if name == "id_all":
        return patch_memory(layers, region, self_states, every)
    if name.startswith(("xq", "xdoc")):
        donor = xq_states if name.startswith("xq") else xdoc_states
        which = every if name.endswith("_all") else [int(name.split("@")[1])]
        return patch_memory(layers, region, donor, which)
    return nothing()


@torch.no_grad()
def blocked_capture(lm, layers, embeds, region, m):
    store = {}
    index = torch.as_tensor(region)
    handles = [layer.register_forward_hook(
        lambda _m, _a, o, l=l: store.__setitem__(
            l, (o[0] if isinstance(o, (tuple, list)) else o)[0, index.to(embeds.device)].clone()))
        for l, layer in enumerate(layers)]
    try:
        mask = block_mask(embeds.size(1), *m, dtype=embeds.dtype, device=embeds.device)
        lm(inputs_embeds=embeds, attention_mask=mask, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return [store[l] for l in range(len(layers))]


# -- summary ------------------------------------------------------------------

def summarize(path):
    rows = [json.loads(l) for l in open(os.path.join(path, "rows.jsonl"), encoding="utf-8")]
    n_pairs = max(r["pair"] for r in rows) + 1
    halves = {"all": rows, "dev": [r for r in rows if r["pair"] < n_pairs / 2],
              "holdout": [r for r in rows if r["pair"] >= n_pairs / 2]}
    names = list(rows[0]["modes"])
    report = {"n_receivers": len(rows), "cells": {}, "vs_none": {}}
    print(f"{path}: {len(rows)} receivers")
    print(f"{'mode':<10}{'NLL':>8}{'sub':>8}{'EM':>8}{'partner_sub':>13}{'same_pred':>11}")
    for name in names:
        x = [r["modes"][name] for r in rows]
        same = np.mean([r["modes"][name]["pred"] == r["modes"]["none"]["pred"] for r in rows])
        cell = {k: float(np.mean([v[k] for v in x])) for k in ("nll", "substring", "em", "partner_substring")}
        cell["same_pred_as_none"] = float(same)
        report["cells"][name] = cell
        print(f"{name:<10}{cell['nll']:>8.4f}{cell['substring']:>8.4f}{cell['em']:>8.4f}"
              f"{cell['partner_substring']:>13.4f}{same:>11.3f}")
    print()
    for name in names[1:]:
        for h, rs in halves.items():
            res = {m: paired_bootstrap([r["modes"][name][m] for r in rs],
                                       [r["modes"]["none"][m] for r in rs], seed=7)
                   for m in ("nll", "substring", "em", "partner_substring")}
            report["vs_none"][f"{name}|{h}"] = res
            print(f"{name:<10}{h:<8}" + "  ".join(
                f"{m} {v['delta']:+.4f} [{v['lo']:+.4f},{v['hi']:+.4f}]" for m, v in res.items()))
    with open(os.path.join(path, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("[summary]", os.path.join(path, "summary.json"))


if __name__ == "__main__":
    args = args_parser()
    if args.summarize:
        summarize(args.summarize)
    else:
        evaluate(args)
