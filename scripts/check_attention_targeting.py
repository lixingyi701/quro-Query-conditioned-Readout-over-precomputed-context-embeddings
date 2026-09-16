"""Does the readout's attention actually land on the document holding the answer?

If a trained readout loses to cosine-similarity top-B, there are two very
different reasons: it produces the wrong *kind* of vector, or it picks the wrong
*documents*.  The distribution statistics rule out the first (after training the
outputs match the cache in norm and sit at cosine ~0.9 to real latents), so this
measures the second directly.

Evaluation rows built with ``--distractors`` record ``gold_rank``: the position of
the genuine passage among the padded retrieval list.  Attention mass falling on
the gold document's ``m`` latents is therefore a clean targeting score, with a
known random baseline of ``1 / K``.

Reported per model:

``gold_attention``    share of attention mass on the gold document's latents
``random_baseline``   n_gold / K, what pure chance would give
``gold_in_topB``      how often the gold document appears among the B most
                      attended latents at all
``cosine_gold``       the same targeting score for non-parametric cosine scoring,
                      i.e. what arm S implicitly achieves
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from src.baselines import pool_query_in_generator_space
from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset, move_to_device, read_jsonl
from src.model import build_model


def gold_positions(row):
    """Document positions holding gold evidence, as an explicit list.

    ``gold_ranks`` is authoritative when present.  The older ``gold_rank`` +
    ``n_gold`` pair describes a contiguous block, which holds for the TriviaQA
    builder but not in general: HotpotQA shuffles its ten paragraphs, and only
    ~21% of its questions end up with their two gold paragraphs adjacent.
    """
    ranks = row.get("gold_ranks")
    if ranks:
        return [int(r) for r in ranks]
    rank = row.get("gold_rank")
    if rank is None:
        return []
    return list(range(int(rank), int(rank) + int(row.get("n_gold", 1))))


def gold_mask_for(rows, batch_ids, m, k, device):
    """(B, K*m) bool marking the latents that belong to the gold document(s)."""
    mask = torch.zeros(len(batch_ids), k * m, dtype=torch.bool)
    for i, row_id in enumerate(batch_ids):
        for j in gold_positions(rows[row_id]):
            if 0 <= j < k:
                mask[i, j * m : (j + 1) * m] = True
    return mask.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pisco_gonogo")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="trivia")
    ap.add_argument("--output_query_mode", default="xattn")
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--rows", type=int, default=256)
    args = ap.parse_args()

    cfg = get_config(args.preset)
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

    path = cfg.data.eval_files[args.split]
    raw = {str(r.get("id")): r for r in read_jsonl(path)[: args.rows]}
    for r in raw.values():
        r.setdefault("n_gold", max(1, len(r["retrieved_doc_ids"]) - r.get("n_distractors", 0)))
    if all(not gold_positions(r) for r in raw.values()):
        raise SystemExit(
            f"{path} marks no gold documents; it needs gold_ranks (HotpotQA) or "
            "gold_rank (rebuild the TriviaQA rows with --distractors)")

    dataset = QuRODataset(path, stack.tokenizer, cfg.data,
                          query_tokenizer=stack.query_tokenizer, limit=args.rows)
    collator = QuROCollator(
        cache, pad_id=model.pad_id,
        query_pad_id=getattr(stack.query_tokenizer, "pad_token_id", model.pad_id),
        max_docs=cfg.data.max_docs)

    learned, cosine, chance, hits, n = [], [], [], [], 0
    m = cache.metadata.latent_size
    with torch.no_grad():
        for start in range(0, len(dataset), 8):
            items = [dataset[i] for i in range(start, min(start + 8, len(dataset)))]
            batch = move_to_device(collator(items), device)
            k = batch["cached_latents"].size(1)
            gold = gold_mask_for(raw, batch["ids"], m, k, device)
            valid = batch["document_mask"][:, :, None].expand(-1, -1, m).reshape(len(items), -1)

            result = model.readout_cached(batch, budget=args.budget, return_attn=True)
            attention = result["aux"]["attention"].mean(1).float()        # (B, budget, K*m)
            share = (attention * gold[:, None, :]).sum(-1)                # (B, budget)
            learned.append(share.mean(-1))

            # What cosine scoring -- arm S's rule -- would have targeted instead.
            memory = batch["cached_latents"].float().reshape(len(items), -1, cache.metadata.hidden_size)
            qvec = pool_query_in_generator_space(
                stack.lm, batch["query_gen_ids"], batch["query_gen_mask"]).float()
            scores = F.cosine_similarity(memory, qvec[:, None, :].expand_as(memory), dim=-1)
            scores = scores.masked_fill(~valid, float("-inf"))
            picked = scores.topk(min(args.budget, scores.size(1)), dim=1).indices
            cosine.append(gold.gather(1, picked).float().mean(-1))

            chance.append(gold.sum(-1).float() / valid.sum(-1).clamp_min(1).float())
            top = attention.mean(1).topk(min(args.budget, attention.size(-1)), dim=-1).indices
            hits.append(gold.gather(1, top).any(-1).float())
            n += len(items)

    report = {
        "checkpoint": os.path.basename(args.checkpoint) if args.checkpoint else "untrained",
        "split": args.split, "rows": n, "budget": args.budget,
        "gold_attention": round(torch.cat(learned).mean().item(), 4),
        "cosine_gold_hit_rate": round(torch.cat(cosine).mean().item(), 4),
        "random_baseline": round(torch.cat(chance).mean().item(), 4),
        "gold_in_topB": round(torch.cat(hits).mean().item(), 4),
    }
    print(json.dumps(report, indent=2))
    print("\ninterpretation:")
    print(f"  learned attention on gold : {report['gold_attention']:.1%}")
    print(f"  cosine selection on gold  : {report['cosine_gold_hit_rate']:.1%}  (arm S's rule)")
    print(f"  chance                    : {report['random_baseline']:.1%}")
    if report["gold_attention"] < report["random_baseline"] * 1.2:
        print("  -> the readout is barely above chance at finding the right document;"
              "\n     it is selecting precisely, but not selecting the right thing")
    elif report["gold_attention"] < report["cosine_gold_hit_rate"]:
        print("  -> the readout targets gold better than chance but worse than cosine,"
              "\n     which is exactly the gap to arm S")
    else:
        print("  -> targeting is not the bottleneck; look elsewhere")


if __name__ == "__main__":
    main()
