"""Executable QuRO v0.0 contract tests (pytest is not required)."""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from src.cache import CacheMetadata, LatentCache, LatentCacheWriter
from src.data import QuROCollator, RAGCompressionDataset
from src.model import build_model


FAILURES = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"{status:4s} {name} {detail}")
    if not condition:
        FAILURES.append(name)


def main():
    torch.manual_seed(0)
    cfg = get_config("tiny")
    cfg.perceiver.num_compressed = 8
    cfg.perceiver.budget_buckets = [2, 4, 8]
    cfg.perceiver.__post_init__()
    cfg.train.out_dir = "runs/_test"
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    tokenizer, generator, model = build_model(cfg)
    model.eval()

    # Two queries, two retrieved document slots; one slot is padding.
    b, k, length = 2, 2, 31
    ids = torch.randint(4, len(tokenizer), (b, k, length))
    token_mask = torch.ones_like(ids, dtype=torch.bool)
    document_mask = torch.tensor([[True, True], [True, False]])
    query_ids = torch.randint(4, len(tokenizer), (b, 7))
    query_mask = torch.ones_like(query_ids, dtype=torch.bool)

    with torch.no_grad():
        per_doc = model.encode_documents(ids, token_mask, document_mask)
        readout = model.readout_cached(
            per_doc, document_mask, query_ids, query_mask, return_attn=True)
    m, d = cfg.perceiver.num_latents, cfg.perceiver.d_latent
    check("independent document encoding", per_doc.shape == (b, k, m, d),
          str(tuple(per_doc.shape)))
    check("online memory is K*m", readout["latent_mask"].shape == (b, k * m))
    check("fixed budget produces B outputs",
          readout["soft_tokens"].shape == (b, 8, model.d_gen))
    check("attention is B by K*m",
          readout["attention"].shape[-2:] == (8, k * m))

    # A masked document must not influence online readout.
    changed = per_doc.clone()
    changed[1, 1] = 1e4
    with torch.no_grad():
        a = model.readout_cached(per_doc, document_mask, query_ids, query_mask)["soft_tokens"]
        c = model.readout_cached(changed, document_mask, query_ids, query_mask)["soft_tokens"]
    check("masked retrieved documents are ignored", (a[1] - c[1]).abs().max() < 1e-5)

    with torch.no_grad():
        shifted = model.readout_cached(
            per_doc, document_mask, query_ids.flip(0), query_mask)["soft_tokens"]
    check("query-conditioned readout changes with query", (a - shifted).abs().mean() > 1e-5)

    cfg_a = get_config("tiny")
    cfg_a.perceiver.output_query_mode = "agnostic"
    cfg_a.perceiver.num_compressed = 8
    cfg_a.perceiver.budget_buckets = [2, 4, 8]
    cfg_a.perceiver.__post_init__()
    cfg_a.train.out_dir = "runs/_test_agnostic"
    _, _, agnostic = build_model(cfg_a)
    agnostic.eval()
    random_latents = torch.randn(b, k, cfg_a.perceiver.num_latents,
                                 cfg_a.perceiver.d_latent)
    with torch.no_grad():
        x = agnostic.readout_cached(
            random_latents, document_mask, query_ids, query_mask)["soft_tokens"]
        y = agnostic.readout_cached(
            random_latents, document_mask, query_ids.flip(0), query_mask)["soft_tokens"]
    check("Ablation A is query agnostic", (x - y).abs().max() < 1e-6)

    with torch.no_grad():
        dynamic = model.readout_cached(
            per_doc, document_mask, query_ids, query_mask, budget=[2, 8])
    check("discrete per-query budgets", dynamic["soft_token_mask"].sum(1).tolist() == [2, 8])

    # Disk cache contract.
    with tempfile.TemporaryDirectory() as directory:
        meta = CacheMetadata("unit-test", m, d, "float32")
        with LatentCacheWriter(directory, meta, shard_size=1) as writer:
            writer.add("d0", per_doc[0, 0], 100)
            writer.add("d1", per_doc[0, 1], 80)
        cache = LatentCache(directory, max_open_shards=1)
        cached, mask, counts = cache.get_many([["d0", "d1"], ["d1"]])
        check("cache round-trip shape", cached.shape == (2, 2, m, d))
        check("cache document mask", mask.tolist() == [[True, True], [True, False]])
        check("cache source token counts", counts.tolist() == [[100, 80], [80, 0]])
        check("cache values round-trip", torch.equal(cached[0, 0], per_doc[0, 0]))

    # Teacher-generated targets take priority over gold answers.
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "rows.jsonl")
        row = {"id": "q0", "query": "question", "retrieved_doc_ids": ["d0"],
               "documents": ["evidence"], "answer": ["gold"],
               "teacher_output": "teacher sequence"}
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        dataset = RAGCompressionDataset(path, tokenizer, cfg.data)
        check("sequence-level KD target selected",
              dataset.rows[0]["target"] == "teacher sequence")
        meta = CacheMetadata("unit-test", m, d, "float32")
        with LatentCacheWriter(os.path.join(directory, "cache"), meta) as writer:
            writer.add("d0", per_doc[0, 0], 12)
        collator = QuROCollator(model.pad_id, model.enc_pad_id,
                               cache=LatentCache(os.path.join(directory, "cache")))
        item = collator([dataset[0]])
        check("collator resolves IDs through cache",
              item["cached_latents"].shape == (1, 1, m, d))

    # End-to-end loss from a dataset row resolved through the cache.
    model.train()
    output = model(item)
    check("cache-first loss is finite", torch.isfinite(output["loss"]).item())
    output["loss"].backward()
    check("gradient reaches readout slots", model.output_query.slots.grad is not None)
    check("frozen generator has no gradients",
          all(parameter.grad is None for parameter in generator.parameters()))

    # Adaptive B is supervised by row-level discrete bucket labels.
    cfg_b = get_config("tiny")
    cfg_b.perceiver.adaptive_budget = True
    cfg_b.perceiver.budget_buckets = [2, 8]
    cfg_b.perceiver.__post_init__()
    cfg_b.train.out_dir = "runs/_test_budget"
    _, _, adaptive = build_model(cfg_b)
    budget_item = {
        "cached_latents": torch.randn(2, 1, cfg_b.perceiver.num_latents,
                                      cfg_b.perceiver.d_latent),
        "document_mask": torch.ones(2, 1, dtype=torch.bool),
        "query_ids": query_ids,
        "query_mask": query_mask,
        "prompt_ids": [[4, 5], [4, 5]],
        "target_ids": [[6, tokenizer.eos_token_id], [7, tokenizer.eos_token_id]],
        "budget": torch.tensor([2, 8]),
    }
    budget_output = adaptive(budget_item)
    check("adaptive budget adds supervised loss", "budget_loss" in budget_output)
    budget_output["loss"].backward()
    check("gradient reaches budget controller",
          any(parameter.grad is not None
              for parameter in adaptive.budget_selector.parameters()))

    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} failures: {FAILURES}")
    print("All QuRO v0.0 contract tests passed.")


if __name__ == "__main__":
    main()
