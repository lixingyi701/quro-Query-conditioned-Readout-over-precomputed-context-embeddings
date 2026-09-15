"""Compare the readout's output distribution against the cached latents it reads.

Two competing explanations for a trained readout losing to non-parametric top-B
selection:

*representational*  the decoder cannot consume attention-weighted mixtures of
    cached latents, so blending is the wrong operation;
*optimisation*      blending is fine in principle -- the decoder LoRA trains, and
    a one-hot attention would reproduce selection exactly -- but the architecture's
    inductive bias is aggregation and the initialisation sits in an averaging basin.

They make different predictions and this script separates them.  If the readout's
outputs have the same norm and per-dimension statistics as real cached latents,
the representational story is dead and the problem is optimisation.  If averaging
has shrunk them (uniform attention over N latents scales the norm by ~1/sqrt(N)
when directions are uncorrelated) and a single learned scalar cannot restore the
direction, the operation itself is suspect.

Reported per model:

``norm_ratio``            ||E|| / mean||Z||, expected ~1 for selection
``max_cos_to_cache``      cosine to the closest latent actually in the batch's memory
``effective_support``     exp(attention entropy): how many latents each output
                          really draws on.  1 = selection, K*m = averaging.
``per_dim_std_ratio``     std(E) / std(Z) per dimension, averaged
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset, move_to_device
from src.model import build_model


def stats_for(model, loader_items, collator, device, budget):
    norms, cosines, supports, e_all, z_all, diversity = [], [], [], [], [], []
    with torch.no_grad():
        for start in range(0, len(loader_items), 8):
            batch = move_to_device(collator(loader_items[start : start + 8]), device)
            result = model.readout_cached(batch, budget=budget, return_attn=True)
            e = result["soft_tokens"].float()                       # (B, budget, h)
            z = batch["cached_latents"].float()
            b, k, m, h = z.shape
            memory = z.reshape(b, k * m, h)
            mask = result["aux"]["latent_mask"]

            norms.append(e.norm(dim=-1).flatten() /
                         (memory.norm(dim=-1) * mask).sum(1).div(mask.sum(1).clamp_min(1))
                         .repeat_interleave(e.size(1)))
            cos = torch.nn.functional.cosine_similarity(
                e[:, :, None, :], memory[:, None, :, :], dim=-1)    # (B, budget, K*m)
            cos = cos.masked_fill(~mask[:, None, :], -1.0)
            cosines.append(cos.max(-1).values.flatten())

            attention = result["aux"].get("attention")
            if attention is not None:
                p = attention.mean(1).float().clamp_min(1e-9)
                supports.append((-(p.log() * p).sum(-1)).exp().flatten())
            # Redundancy check: a prior shared by every slot can make all B
            # outputs the same vector, which wastes B-1 of the budget.
            unit = torch.nn.functional.normalize(e, dim=-1)
            pair = unit @ unit.transpose(1, 2)                   # (B, budget, budget)
            off = ~torch.eye(e.size(1), dtype=torch.bool, device=e.device)
            diversity.append(pair[:, off].mean(-1))
            e_all.append(e.reshape(-1, h).cpu())
            z_all.append(memory[mask].cpu())

    e_cat, z_cat = torch.cat(e_all), torch.cat(z_all)

    out = {
        "norm_ratio": round(torch.cat(norms).mean().item(), 4),
        "max_cos_to_cache": round(torch.cat(cosines).mean().item(), 4),
        "per_dim_std_ratio": round(
            (e_cat.std(0) / z_cat.std(0).clamp_min(1e-6)).mean().item(), 4),
        "output_norm": round(e_cat.norm(dim=-1).mean().item(), 3),
        "cache_norm": round(z_cat.norm(dim=-1).mean().item(), 3),
    }
    if supports:
        out["effective_support"] = round(torch.cat(supports).mean().item(), 2)
    if diversity:
        out["inter_slot_cosine"] = round(torch.cat(diversity).mean().item(), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pisco_gonogo")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--readout", default="quro",
                    choices=["quro", "pisco_direct", "similarity_topb"])
    ap.add_argument("--output_query_mode", default="xattn")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--rows", type=int, default=64)
    args = ap.parse_args()

    cfg = get_config(args.preset)
    cfg.readout.kind = args.readout
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
        _, _, step = model.load(args.checkpoint)
        print(f"loaded {os.path.basename(args.checkpoint)} (step {step})")
    model.eval()

    dataset = QuRODataset(cfg.data.eval_files["trivia"], stack.tokenizer, cfg.data,
                          query_tokenizer=stack.query_tokenizer, limit=args.rows)
    collator = QuROCollator(
        cache, pad_id=model.pad_id,
        query_pad_id=getattr(stack.query_tokenizer, "pad_token_id", model.pad_id),
        max_docs=cfg.data.max_docs)
    items = [dataset[i] for i in range(len(dataset))]

    report = stats_for(model, items, collator, device, args.budget)
    report["readout"] = args.readout
    report["checkpoint"] = os.path.basename(args.checkpoint) if args.checkpoint else "untrained"
    print(json.dumps(report, indent=2))

    print("\ninterpretation:")
    ratio = report["norm_ratio"]
    print(f"  ||E|| / mean||Z|| = {ratio:.3f}"
          + ("  -> matches the cache scale" if 0.8 <= ratio <= 1.25
             else "  -> the output lives at a different scale than any cached latent"))
    cos = report["max_cos_to_cache"]
    print(f"  best cosine to any latent in memory = {cos:.3f}"
          + ("  -> essentially selecting" if cos > 0.9
             else "  -> a mixture, not any single cached vector"))
    if "effective_support" in report:
        print(f"  effective support = {report['effective_support']:.1f} latents per output"
              " (1 = selection)")
    if "inter_slot_cosine" in report:
        value = report["inter_slot_cosine"]
        print(f"  mean cosine between output slots = {value:.3f}"
              + ("  -> the slots are near-duplicates; B-1 of the budget is wasted"
                 if value > 0.9 else "  -> the slots carry different evidence"))


if __name__ == "__main__":
    main()
