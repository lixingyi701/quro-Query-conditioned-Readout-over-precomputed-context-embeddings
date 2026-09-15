"""Executable contract tests for QuRO v0.1 -- CPU only, no downloads.

These cover the invariants that experiments silently depend on: readout shapes,
the residual initialisation that makes step 0 equal attention-pooled PISCO, the
nested budget structure, the PISCO-identical prompt slots, and end-to-end
gradient flow.  Run with ``python tests/test_shapes.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from src.baselines import PiscoDirectReadout, SimilarityTopBReadout
from src.cache import CacheMetadata, LatentCache, LatentCacheWriter
from src.data import QuROCollator, QuRODataset, doc_id_for, move_to_device
from src.model import build_model
from src.prompt import PiscoPromptBuilder
from src.readout import QuroReadout

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"{'PASS' if condition else 'FAIL'} {name} {detail}")


# ----------------------------------------------------------------------------
# Synthetic workspace: a tiny corpus, its latent cache, and matching query rows.
# ----------------------------------------------------------------------------
def build_workspace(root, num_docs=12, m=4, h=64, num_rows=8):
    texts = [f"Document number {i} states that the code word is alpha{i} and nothing else."
             for i in range(num_docs)]
    doc_ids = [doc_id_for(t) for t in texts]

    torch.manual_seed(0)
    cache_dir = os.path.join(root, "cache")
    metadata = CacheMetadata(compressor="synthetic", latent_size=m, hidden_size=h,
                             dtype="float32", compr_rate=16, doc_max_length=128)
    with LatentCacheWriter(cache_dir, metadata, shard_size=5) as writer:
        for i, doc_id in enumerate(doc_ids):
            writer.add(doc_id, torch.randn(m, h), source_token_count=100 + i,
                       original_token_count=150 + i)

    rows = []
    for i in range(num_rows):
        picked = [doc_ids[i % num_docs], doc_ids[(i + 1) % num_docs]]
        rows.append({"id": f"q{i}", "query": f"What is the code word in document {i}?",
                     "retrieved_doc_ids": picked, "answers": [f"alpha{i % num_docs}"]})
    train_path = os.path.join(root, "train.jsonl")
    with open(train_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return cache_dir, train_path, doc_ids, m, h


# ----------------------------------------------------------------------------
def test_readout(m=4, h=64, cache_h=64):
    torch.manual_seed(0)
    b, k, budget, q_dim = 3, 2, 8, 32
    readout = QuroReadout(cache_hidden=cache_h, gen_hidden=cache_h, query_dim=q_dim,
                          d_readout=48, max_budget=budget, num_blocks=2, num_heads=4)
    latents = torch.randn(b, k, m, cache_h)
    doc_mask = torch.ones(b, k, dtype=torch.bool)
    doc_mask[2, 1] = False
    query = torch.randn(b, 6, q_dim)
    query_mask = torch.ones(b, 6, dtype=torch.bool)

    out, aux = readout(latents, doc_mask, query, query_mask, budget=budget, return_attn=True)
    check("readout shape is (B, budget, h_gen)", tuple(out.shape) == (b, budget, cache_h),
          str(tuple(out.shape)))
    check("attention is (B, H, budget, K*m)",
          tuple(aux["attention"].shape)[-2:] == (budget, k * m))
    check("masked documents get zero attention",
          float(aux["attention"][2, :, :, m:].abs().max()) < 1e-6)

    # Zero-initialised residual: step 0 must equal attention-pooled cached latents.
    alpha = aux["attention"].mean(1)
    pooled = torch.bmm(alpha, latents.reshape(b, k * m, cache_h))
    check("residual init reproduces attention-pooled latents",
          torch.allclose(out, pooled, atol=1e-4),
          f"max|diff|={float((out - pooled).abs().max()):.2e}")

    smaller, _ = readout(latents, doc_mask, query, query_mask, budget=4)
    check("smaller budget is a prefix of the larger one",
          torch.allclose(smaller, out[:, :4], atol=1e-4))

    other = torch.randn(b, 6, q_dim)
    shifted, _ = readout(latents, doc_mask, other, query_mask, budget=budget)
    check("readout responds to the query", not torch.allclose(shifted, out, atol=1e-3))

    agnostic = QuroReadout(cache_hidden=cache_h, gen_hidden=cache_h, query_dim=q_dim,
                           d_readout=48, max_budget=budget, num_heads=4,
                           output_query_mode="agnostic")
    a1, _ = agnostic(latents, doc_mask, query, query_mask, budget=budget)
    a2, _ = agnostic(latents, doc_mask, other, query_mask, budget=budget)
    check("ablation A ignores the query (variant A)", torch.allclose(a1, a2, atol=1e-6))

    params = sum(p.numel() for p in readout.parameters())
    check("readout parameter count is reported", params > 0, f"{params/1e6:.3f}M at d_r=48")


def test_baselines(m=4, h=64):
    torch.manual_seed(0)
    b, k = 2, 3
    latents = torch.randn(b, k, m, h)
    doc_mask = torch.ones(b, k, dtype=torch.bool)
    doc_mask[1, 2] = False

    direct = PiscoDirectReadout(h, h)
    out, aux = direct(latents, doc_mask)
    check("pisco_direct emits K*m tokens", tuple(out.shape) == (b, k * m, h))
    check("pisco_direct budget follows the document mask",
          int(aux["token_mask"][1].sum()) == 2 * m)

    topb = SimilarityTopBReadout(h, h)
    out, aux = topb(latents, doc_mask, budget=5, query_vector=torch.randn(b, h))
    check("similarity_topb emits exactly B tokens", tuple(out.shape) == (b, 5, h))
    check("similarity_topb never selects a masked document",
          bool(aux["token_mask"].all()))


def test_prompt(tokenizer, n_mem_tokens):
    d0 = PiscoPromptBuilder(tokenizer, n_mem_tokens, "D0")
    d1 = PiscoPromptBuilder(tokenizer, n_mem_tokens, "D1")
    query = "who designed the building and in which year"
    for budget in (1, 4, 8, 16):
        built = d0.build(query, budget)
        check(f"prompt exposes exactly {budget} memory slots",
              len(built.slot_positions) == budget)
    a, b = d0.build(query, 8), d1.build(query, 8)
    check("D1 keeps the slots but drops the question text",
          len(b.slot_positions) == 8 and len(b.input_ids) < len(a.input_ids))
    check("D1 removes every query token",
          not set(tokenizer(query)["input_ids"]).issubset(set(b.input_ids)))


def test_cache(cache_dir, doc_ids, m, h):
    cache = LatentCache(cache_dir)
    check("cache manifest records the compression rate", cache.metadata.compr_rate == 16)
    check("cache manifest records doc_max_length", cache.metadata.doc_max_length == 128)
    latents, doc_mask, counts = cache.get_many([[doc_ids[0], doc_ids[1]], [doc_ids[2]]])
    check("cache batches to (B, K, m, h)", tuple(latents.shape) == (2, 2, m, h))
    check("padded document slots are masked", doc_mask.tolist() == [[True, True], [True, False]])
    check("source token counts come back", counts[0, 0].item() == 100)
    try:
        cache.get_many([["missing-doc"]])
        check("unknown document IDs raise", False)
    except KeyError:
        check("unknown document IDs raise", True)


def test_end_to_end(cache_dir, train_path, m, h):
    cfg = get_config("toy")
    with tempfile.TemporaryDirectory() as out_dir:
        cfg.train.out_dir = out_dir
        cfg.data.train_file = train_path
        cfg.data.cache_dir = cache_dir
        cfg.data.max_docs = 2
        cfg.readout.cache_hidden = h
        cfg.readout.max_budget = 8
        cfg.readout.budget_buckets = [4, 8]
        cfg.generator.toy_d_model = h
        cfg.revalidate()

        cache = LatentCache(cache_dir)
        stack, model = build_model(cfg, cache_hidden=h)
        collator = QuROCollator(cache, pad_id=model.pad_id, max_docs=2)
        dataset = QuRODataset(train_path, stack.tokenizer, cfg.data,
                              query_tokenizer=stack.query_tokenizer)
        batch = collator([dataset[i] for i in range(4)])

        result = model.readout_cached(batch, budget=8)
        check("end-to-end soft tokens are (B, budget, d_gen)",
              tuple(result["soft_tokens"].shape) == (4, 8, h))

        output = model(batch, budget=8, residual_weight=0.1)
        check("loss is finite", torch.isfinite(output["loss"]).item())
        check("residual penalty is reported", "residual_penalty" in output)

        output["loss"].backward()
        slots = model.readout.output_query.slots
        check("gradient reaches the output query slots",
              slots.grad is not None and float(slots.grad.abs().sum()) > 0)
        # The residual branch is zero-initialised, so any norm taken with sqrt()
        # has an infinite derivative at step 0 and silently NaNs every gradient.
        bad = [name for name, p in model.named_parameters()
               if p.grad is not None and not torch.isfinite(p.grad).all()]
        check("every gradient is finite at the zero-residual initialisation",
              not bad, f"non-finite: {bad[:3]}")

        predictions = model.generate_answer(batch, max_new_tokens=4)
        check("generation returns one string per row",
              len(predictions) == 4 and all(isinstance(x, str) for x in predictions))

        report = model.parameter_report()
        check("parameter report totals up",
              report["total"] >= report["readout"], json.dumps(report))


def main():
    with tempfile.TemporaryDirectory() as root:
        cache_dir, train_path, doc_ids, m, h = build_workspace(root)
        test_readout(m, h, h)
        test_baselines(m, h)
        test_cache(cache_dir, doc_ids, m, h)

        from src.toy import ToyTokenizer
        test_prompt(ToyTokenizer.build_from_texts(["who designed the building and in which year"]), 8)

        test_end_to_end(cache_dir, train_path, m, h)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
