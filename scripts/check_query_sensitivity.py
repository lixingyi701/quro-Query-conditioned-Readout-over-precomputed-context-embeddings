"""Measure how much the readout's output actually depends on the query.

The whole method rests on one claim: the same cached latents should produce
different soft tokens for different questions.  End-task metrics cannot separate
"the readout selects well" from "the decoder reads the plain-text question and
does the selection itself", so this measures the readout in isolation.

For each example the same documents are read with the real query and with a
mismatched one, and the outputs are compared:

``delta_ratio``
    ``||E(q) - E(q')|| / ||E(q)||``.  Near 0 means the readout is effectively
    query-agnostic no matter what the end metrics say.
``attention_tvd``
    Total variation distance between the two attention distributions over the
    ``K*m`` cached latents -- does the readout even look somewhere else?
``attention_entropy``
    In nats, against ``log(K*m)``.  A value near the maximum means attention is
    uniform, i.e. the readout is averaging rather than selecting.

Run it against an untrained model to confirm the wiring, and against a trained
checkpoint to see whether training moved the numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset, move_to_device
from src.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pisco_smoke")
    ap.add_argument("--checkpoint", default=None, help="omit to probe an untrained model")
    ap.add_argument("--output_query_mode", default="xattn")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_file", default=None,
                    help="override the preset's dev split (measure each model on its own data)")
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = get_config(args.preset)
    if args.eval_file:
        cfg.data.eval_files = {"dev": args.eval_file}
    if args.max_docs is not None:
        cfg.data.max_docs = args.max_docs
    cfg.readout.output_query_mode = args.output_query_mode
    cfg.generator.lora_init = "frozen"
    cfg.revalidate()

    cache = LatentCache(cfg.data.cache_dir)
    cfg.readout.cache_hidden = cache.metadata.hidden_size
    stack, model = build_model(cfg, cache_hidden=cache.metadata.hidden_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    stack.lm.to(device)
    if args.checkpoint:
        missing, unexpected, step = model.load(args.checkpoint)
        print(f"loaded {args.checkpoint} (step {step}, {len(missing)} missing, "
              f"{len(unexpected)} unexpected)")
    model.eval()

    dataset = QuRODataset(cfg.data.eval_files["dev"], stack.tokenizer, cfg.data,
                          query_tokenizer=stack.query_tokenizer, limit=args.rows)
    shifted = QuRODataset(cfg.data.eval_files["dev"], stack.tokenizer, cfg.data,
                          query_tokenizer=stack.query_tokenizer, query_shift=1,
                          limit=args.rows)
    collator = QuROCollator(cache, pad_id=model.pad_id,
                            query_pad_id=getattr(stack.query_tokenizer, "pad_token_id", model.pad_id),
                            max_docs=cfg.data.max_docs)

    deltas, tvds, entropies, max_entropies = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(dataset), args.batch_size):
            index = range(start, min(start + args.batch_size, len(dataset)))
            real = move_to_device(collator([dataset[i] for i in index]), device)
            fake = move_to_device(collator([shifted[i] for i in index]), device)

            a = model.readout_cached(real, budget=args.budget, return_attn=True)
            b = model.readout_cached(fake, budget=args.budget, return_attn=True)
            ea, eb = a["soft_tokens"].float(), b["soft_tokens"].float()
            deltas.append(((ea - eb).norm(dim=-1) / ea.norm(dim=-1).clamp_min(1e-6)).flatten())

            if a["aux"].get("attention") is None:
                continue
            pa = a["aux"]["attention"].mean(1).float()       # (B, budget, K*m)
            pb = b["aux"]["attention"].mean(1).float()
            tvds.append((0.5 * (pa - pb).abs().sum(-1)).flatten())
            entropies.append((-(pa.clamp_min(1e-9).log() * pa).sum(-1)).flatten())
            valid = a["aux"]["latent_mask"].sum(-1).clamp_min(1).float()
            max_entropies.append(valid.log()[:, None].expand(-1, pa.size(1)).flatten())

    def stats(values):
        x = torch.cat(values)
        return {"mean": round(x.mean().item(), 4), "median": round(x.median().item(), 4),
                "p90": round(x.quantile(0.9).item(), 4)}

    report = {
        "checkpoint": args.checkpoint or "untrained",
        "output_query_mode": args.output_query_mode,
        "budget": args.budget,
        "rows": len(dataset),
        "delta_ratio": stats(deltas),
    }
    if tvds:
        entropy = torch.cat(entropies)
        ceiling = torch.cat(max_entropies)
        report["attention_tvd"] = stats(tvds)
        report["attention_entropy_nats"] = stats(entropies)
        report["attention_entropy_max_nats"] = round(ceiling.mean().item(), 4)
        report["attention_entropy_fraction_of_max"] = round(
            (entropy / ceiling.clamp_min(1e-6)).mean().item(), 4)
    print(json.dumps(report, indent=2))

    delta = report["delta_ratio"]["median"]
    print("\ninterpretation:")
    print(f"  median ||E(q)-E(q')|| / ||E(q)|| = {delta:.3f}")
    if delta < 0.02:
        print("  -> the readout is effectively query-agnostic; any end-task query")
        print("     sensitivity is coming from the plain-text question in the prompt")
    elif delta < 0.15:
        print("  -> the readout responds to the query but weakly")
    else:
        print("  -> the readout produces substantially different evidence per query")
    if "attention_entropy_fraction_of_max" in report:
        frac = report["attention_entropy_fraction_of_max"]
        print(f"  attention entropy is {frac:.1%} of uniform"
              + ("  -> averaging, not selecting" if frac > 0.95 else "  -> genuinely peaked"))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
