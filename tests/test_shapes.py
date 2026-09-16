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


def test_arms(m=4, h=64, cache_h=64):
    """Query-invariance of the A arms, with the cosine prior actually supplied.

    The pre-existing agnostic test above passes ``query_vector=None``, so it never
    exercised ``cosine_bias`` -- which is the second, independent route by which
    the query reaches the readout.  Every A run on disk had ``cosine_prior=True``,
    so it was A1 and not the query-agnostic control it was reported as
    (docs/warning_and_target.md W1).  These checks make that unstatable.
    """
    torch.manual_seed(0)
    b, k, budget, q_dim = 3, 2, 8, 32
    latents = torch.randn(b, k, m, cache_h)
    doc_mask = torch.ones(b, k, dtype=torch.bool)
    doc_mask[2, 1] = False
    query = torch.randn(b, 6, q_dim)
    query_mask = torch.ones(b, 6, dtype=torch.bool)
    other = torch.randn(b, 6, q_dim)
    # The prior lives in the cache's space, not the query encoder's.
    vector, other_vector = torch.randn(b, cache_h), torch.randn(b, cache_h)

    def build(output_query_mode, cosine_prior):
        torch.manual_seed(1)
        return QuroReadout(cache_hidden=cache_h, gen_hidden=cache_h, query_dim=q_dim,
                           d_readout=48, max_budget=budget, num_heads=4,
                           output_query_mode=output_query_mode, cosine_prior=cosine_prior)

    def swap_query(readout, **kw):
        a, _ = readout(latents, doc_mask, query, query_mask, budget=budget,
                       query_vector=vector, **kw)
        c, _ = readout(latents, doc_mask, other, query_mask, budget=budget,
                       query_vector=other_vector, **kw)
        return a, c

    for arm, mode, prior, invariant in (
        ("A0", "agnostic", False, True),
        ("A0m", "agnostic_matched", False, True),
        ("A1", "agnostic", True, False),
        ("C0", "xattn", False, False),
        ("C1", "xattn", True, False),
    ):
        a, c = swap_query(build(mode, prior))
        same = torch.allclose(a, c, atol=1e-6)
        check(f"arm {arm} is {'invariant to' if invariant else 'conditioned on'} the query",
              same == invariant, f"max|diff|={float((a - c).abs().max()):.2e}")

    # A1 was the historical "query-agnostic" control; name the route it actually uses.
    a1 = build("agnostic", True)
    fixed, _ = a1(latents, doc_mask, query, query_mask, budget=budget, query_vector=vector)
    no_prior, _ = a1(latents, doc_mask, other, query_mask, budget=budget, query_vector=None)
    check("A1's query dependence is entirely the cosine prior",
          not torch.allclose(fixed, no_prior, atol=1e-6))

    # Parameter matching: plain agnostic drops the query cross-attention block, so
    # it is not a like-for-like control for C on parameter count.
    n = {mode: sum(p.numel() for p in build(mode, True).parameters())
         for mode in ("agnostic", "agnostic_matched", "xattn")}
    check("plain agnostic is NOT parameter-matched to C", n["agnostic"] < n["xattn"],
          f"{n['agnostic']} vs {n['xattn']}")
    check("agnostic_matched is parameter-matched to C within the placeholder",
          n["agnostic_matched"] > n["agnostic"]
          and abs(n["agnostic_matched"] - n["xattn"]) <= 16 * q_dim,
          f"{n['agnostic_matched']} vs {n['xattn']}")


def test_output_modes(m=4, h=64, cache_h=64):
    """full / pool_only / delta_only must compose exactly, and be probe-able."""
    torch.manual_seed(0)
    b, k, budget, q_dim = 3, 2, 8, 32
    latents = torch.randn(b, k, m, cache_h)
    doc_mask = torch.ones(b, k, dtype=torch.bool)
    query = torch.randn(b, 6, q_dim)
    query_mask = torch.ones(b, 6, dtype=torch.bool)

    torch.manual_seed(1)
    readout = QuroReadout(cache_hidden=cache_h, gen_hidden=cache_h, query_dim=q_dim,
                          d_readout=48, max_budget=budget, num_heads=4)
    # Make Delta non-zero, otherwise full and pool_only agree trivially.
    with torch.no_grad():
        readout.out_proj.weight.normal_(std=0.02)
        readout.out_proj.bias.normal_(std=0.02)

    def run(mode):
        out, aux = readout(latents, doc_mask, query, query_mask, budget=budget,
                           output_mode=mode)
        return out, aux

    full, aux_full = run("full")
    pool, _ = run("pool_only")
    delta, _ = run("delta_only")
    check("full == pool_only + delta_only", torch.allclose(full, pool + delta, atol=1e-5),
          f"max|diff|={float((full - pool - delta).abs().max()):.2e}")
    check("the branches are not degenerate",
          delta.abs().max() > 1e-4 and pool.abs().max() > 1e-4)
    check("aux records the branch actually taken", aux_full["output_mode"] == "full")

    # The legacy flag changed the branch *and* the initialisation together, which is
    # what made its result unattributable.  They are separate knobs now.
    torch.manual_seed(1)
    legacy = QuroReadout(cache_hidden=cache_h, gen_hidden=cache_h, query_dim=q_dim,
                         d_readout=48, max_budget=budget, num_heads=4,
                         residual_readout=False)
    check("legacy residual_readout=False maps to delta_only",
          legacy.output_mode == "delta_only" and legacy.out_proj_init == "default")
    torch.manual_seed(1)
    matched = QuroReadout(cache_hidden=cache_h, gen_hidden=cache_h, query_dim=q_dim,
                          d_readout=48, max_budget=budget, num_heads=4,
                          output_mode="delta_only", out_proj_init="zeros")
    check("delta_only can keep the zero init, so init is separable from the branch",
          float(matched.out_proj.weight.abs().max()) == 0.0)

    # pool_only must not leave a residual penalty pointing at a dead branch.
    _, aux_pool = run("pool_only")
    check("pool_only reports a zero delta_ms", float(aux_pool["delta_ms"]) == 0.0)
    _, aux_delta = run("delta_only")
    check("delta_only reports a zero pooled_ms", float(aux_delta["pooled_ms"]) == 0.0)


def test_query_path_separation(cache_dir, train_path, m, h):
    """The readout's question and the decoder's must be shiftable independently.

    The old ``query_shift`` moved both at once, so a drop under the mismatch
    control could not be attributed to the readout (warning_and_target.md W4).
    """
    from src.toy import ToyTokenizer

    cfg = get_config("toy")
    # Built from the rows' own text: a vocabulary that maps every question to the
    # same UNK sequence would make the tokenised paths look identical whether or
    # not they were actually separated.
    with open(train_path, encoding="utf-8") as f:
        queries = [json.loads(line)["query"] for line in f]
    tokenizer = ToyTokenizer.build_from_texts(queries)

    def rows_for(**shifts):
        data = QuRODataset(train_path, tokenizer, cfg.data, **shifts)
        return [data[i] for i in range(len(data))]

    base = rows_for()
    readout_only = rows_for(readout_query_shift=1, decoder_query_shift=0)
    decoder_only = rows_for(readout_query_shift=0, decoder_query_shift=1)
    legacy = rows_for(query_shift=1)

    check("readout-only shift leaves the decoder's question intact",
          all(a["query"] == b["query"] for a, b in zip(base, readout_only)))
    check("readout-only shift does move the readout's question",
          any(a["readout_query"] != b["readout_query"]
              for a, b in zip(base, readout_only)))
    check("readout-only shift moves the tokenised readout input",
          any(a["query_ids"] != b["query_ids"] for a, b in zip(base, readout_only)))
    check("decoder-only shift leaves the readout's input intact",
          all(a["query_ids"] == b["query_ids"] for a, b in zip(base, decoder_only))
          and all(a["query_gen_ids"] == b["query_gen_ids"]
                  for a, b in zip(base, decoder_only)))
    check("decoder-only shift does move the prompt's question",
          any(a["query"] != b["query"] for a, b in zip(base, decoder_only)))
    check("legacy query_shift still moves both routes together",
          all(a["query"] == a["readout_query"] for a in legacy)
          and any(a["query"] != b["query"] for a, b in zip(base, legacy)))
    check("the swapped question's answers are carried for collision checking",
          all("readout_query_answers" in r and r["readout_query_answers"]
              for r in readout_only))


def test_arm_labels():
    """A config must report the arm it implements, not the arm it is tagged."""
    from config import apply_arm, arm_label, get_config

    for arm in ("A0", "A1", "C0", "C1", "S", "P"):
        cfg = apply_arm(get_config("toy"), arm)
        check(f"apply_arm({arm}) round-trips through arm_label", arm_label(cfg) == arm,
              f"got {arm_label(cfg)}")

    # The historical failure mode: launched as A, but the prior left on.
    cfg = get_config("toy")
    cfg.readout.output_query_mode = "agnostic"
    cfg.readout.cosine_prior = True
    cfg.revalidate()
    check("an A arm with the cosine prior still on is labelled A1", arm_label(cfg) == "A1")

    cfg = apply_arm(get_config("toy"), "A0", param_matched=True)
    check("param-matched A0 is labelled distinctly", arm_label(cfg) == "A0m",
          f"got {arm_label(cfg)}")


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
        test_arms(m, h, h)
        test_output_modes(m, h, h)
        test_arm_labels()
        test_query_path_separation(cache_dir, train_path, m, h)
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
