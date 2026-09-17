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
    (docs/HANDOFF.md §3 W1).  These checks make that unstatable.
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


def test_cosine_prior_short_rows(m=4, h=64, cache_h=64):
    """A row with fewer valid latents than the budget must not poison the batch.

    ``cosine_bias`` ranks candidates with ``topk`` over scores whose invalid
    entries are -inf.  When a row holds fewer valid latents than there are slots,
    topk runs off the end of the mask and the surplus centres come back -inf,
    which makes every ``|score - centre|`` infinite: the slot's whole attention
    row is -inf, its softmax is NaN, and the backward pass spreads that to every
    parameter.  HotpotQA hit this at B=32, where 86 training questions carry only
    two paragraphs; B=8 and B=16 stayed under the limit and hid it.
    """
    torch.manual_seed(0)
    b, k, budget, q_dim = 3, 4, 8, 32
    latents = torch.randn(b, k, m, cache_h)
    doc_mask = torch.ones(b, k, dtype=torch.bool)
    # Row 0 keeps one document: m=4 valid latents against a budget of 8.
    doc_mask[0, 1:] = False
    doc_mask[1, 2:] = False
    query = torch.randn(b, 6, q_dim)
    query_mask = torch.ones(b, 6, dtype=torch.bool)
    vector = torch.randn(b, cache_h)

    readout = QuroReadout(cache_hidden=cache_h, gen_hidden=cache_h, query_dim=q_dim,
                          d_readout=48, max_budget=budget, num_heads=4,
                          cosine_prior=True)
    out, aux = readout(latents, doc_mask, query, query_mask, budget=budget,
                       query_vector=vector, return_attn=True)
    check("short rows do not produce NaN outputs", torch.isfinite(out).all().item(),
          f"{int((~torch.isfinite(out)).sum())} non-finite")
    check("short rows do not produce NaN attention",
          torch.isfinite(aux["attention"]).all().item())
    check("masked documents still get zero attention in short rows",
          float(aux["attention"][0, :, :, m:].abs().max()) < 1e-6)

    out.pow(2).mean().backward()
    grads = [p.grad for p in readout.parameters() if p.grad is not None]
    check("gradients stay finite with short rows",
          all(torch.isfinite(g).all().item() for g in grads),
          f"{len(grads)} gradients")

    # The budget must also be allowed to exceed the number of valid latents.
    wide, _ = readout(latents, doc_mask, query, query_mask, budget=budget,
                      query_vector=vector)
    check("a budget above the valid-latent count is finite",
          torch.isfinite(wide).all().item())


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
    control could not be attributed to the readout (HANDOFF.md §3 W4).
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


def test_query_representation():
    """W3: ``fixed_adapter`` must make the query a genuinely fixed function.

    With ``kind="generator"`` the query encoder and the decoder are one object, so
    ``no_grad`` stops gradient but not drift: training the decoder's LoRA changes
    what the query encodes into.  ``fixed_adapter`` encodes the query through a
    frozen copy taken at init instead.

    Switching adapters is the risky part.  PEFT documents that ``set_adapter``
    sets the target adapter to ``requires_grad=True``, so a careless switch hands
    the optimiser a different trainable set -- and because the optimiser holds the
    Parameter objects, that shows up as the decoder silently not training rather
    than as an error.  These checks cover the four acceptance criteria in
    docs/TRAINING_STRATEGY_REVIEW_AND_PLAN.md §5.2.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError:
        check("query representation (peft unavailable, skipped)", True)
        return

    from types import SimpleNamespace

    from src.model import GeneratorQueryEncoder

    from transformers import PretrainedConfig

    class TinyLM(torch.nn.Module):
        """Smallest object satisfying the LM interface the query path uses.

        ``config`` has to be a real ``PretrainedConfig``: PEFT probes it with
        ``.get``, so a plain namespace fails at injection time.
        """

        def __init__(self, vocab=32, dim=16):
            super().__init__()
            self.config = PretrainedConfig(hidden_size=dim)
            self.embed = torch.nn.Embedding(vocab, dim)
            self.q_proj = torch.nn.Linear(dim, dim)

        def forward(self, input_ids=None, attention_mask=None,
                    output_hidden_states=False, **_):
            hidden = self.q_proj(self.embed(input_ids))
            return SimpleNamespace(hidden_states=(hidden,), last_hidden_state=hidden)

    torch.manual_seed(0)
    base = TinyLM()
    lm = get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj"],
                                         lora_alpha=8, lora_dropout=0.0),
                        adapter_name="decoder_adapter")
    for name, parameter in lm.named_parameters():
        parameter.requires_grad_("lora_" in name and "decoder_adapter" in name)
    trainable_before = {n for n, p in lm.named_parameters() if p.requires_grad}

    encoder = GeneratorQueryEncoder(lm, pooling="mean", representation="fixed_adapter",
                                    adapter_name="decoder_adapter")
    check("fixed_adapter creates a frozen query adapter",
          encoder.query_adapter is not None and encoder.query_adapter_hash is not None)
    check("the frozen copy is not trainable",
          not any(p.requires_grad for n, p in lm.named_parameters()
                  if encoder.query_adapter in n))
    check("creating the copy leaves the decoder's trainable set unchanged",
          {n for n, p in lm.named_parameters() if p.requires_grad} == trainable_before,
          f"{len(trainable_before)} tensors")

    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)
    lm.train()
    before = encoder(ids, mask).clone()
    check("adapter and mode are restored after a query forward",
          lm.training and encoder._active_adapters(lm)[0] == "decoder_adapter")
    check("requires_grad is restored after a query forward",
          {n for n, p in lm.named_parameters() if p.requires_grad} == trainable_before)

    # Criterion 1/2: train the decoder adapter, then re-encode the same query.
    optimiser = torch.optim.SGD([p for p in lm.parameters() if p.requires_grad], lr=0.5)
    for _ in range(5):
        optimiser.zero_grad()
        lm(input_ids=ids, attention_mask=mask.long(),
           output_hidden_states=True).hidden_states[-1].pow(2).mean().backward()
        optimiser.step()
    moved = any(p.grad is not None and float(p.grad.abs().max()) > 0
                for n, p in lm.named_parameters() if p.requires_grad)
    check("the decoder adapter really did update", moved)

    after = encoder(ids, mask)
    check("fixed_adapter: the query representation is unchanged by decoder training",
          torch.allclose(before, after, atol=1e-6),
          f"max|diff|={float((before - after).abs().max()):.2e}")
    check("the frozen adapter's hash is unchanged",
          GeneratorQueryEncoder.adapter_hash(lm, encoder.query_adapter)
          == encoder.query_adapter_hash)

    # The control: shared_current must drift, or the comparison is vacuous.
    torch.manual_seed(0)
    base2 = TinyLM()
    lm2 = get_peft_model(base2, LoraConfig(r=4, target_modules=["q_proj"],
                                           lora_alpha=8, lora_dropout=0.0),
                         adapter_name="decoder_adapter")
    for name, parameter in lm2.named_parameters():
        parameter.requires_grad_("lora_" in name and "decoder_adapter" in name)
    shared = GeneratorQueryEncoder(lm2, pooling="mean", representation="shared_current",
                                   adapter_name="decoder_adapter")
    check("shared_current creates no extra adapter", shared.query_adapter is None)
    drift_before = shared(ids, mask).clone()
    optimiser2 = torch.optim.SGD([p for p in lm2.parameters() if p.requires_grad], lr=0.5)
    for _ in range(5):
        optimiser2.zero_grad()
        lm2(input_ids=ids, attention_mask=mask.long(),
            output_hidden_states=True).hidden_states[-1].pow(2).mean().backward()
        optimiser2.step()
    drift_after = shared(ids, mask)
    check("shared_current: the query representation DOES drift with the decoder",
          not torch.allclose(drift_before, drift_after, atol=1e-6),
          f"max|diff|={float((drift_before - drift_after).abs().max()):.2e}")

    # A1 and S never call forward() -- needs_query is False -- and used to reach
    # the raw LM instead, skipping both the adapter switch and the eval guard.
    # pooled() is the entry point they must come through.
    lm.train()
    pooled_train = encoder.pooled(ids, mask)
    check("pooled() applies the same controls as forward()",
          torch.allclose(pooled_train, encoder.pooled(ids, mask), atol=1e-7)
          and lm.training
          and encoder._active_adapters(lm)[0] == "decoder_adapter")
    check("pooled() agrees with forward()'s pooled vector",
          torch.allclose(pooled_train, encoder.last_pooled, atol=1e-7))


def test_query_vector_routing():
    """A1 and S must not reach the raw LM behind the representation strategy.

    Both have ``needs_query=False``, so ``readout_cached`` never calls the
    encoder's forward, and the old fallback used
    ``pool_query_in_generator_space`` on the raw LM -- skipping the adapter switch
    and the eval-mode guard.  The two arms C1's margin is measured against were
    therefore encoding queries differently from C1 while the run record said
    ``fixed_adapter``, so the bypass biased the comparison rather than merely
    mislabelling it.
    """
    from types import SimpleNamespace

    from src import model as model_module
    from src.model import QuROModel

    raw_calls = {"n": 0}
    original = model_module.pool_query_in_generator_space

    def counting(*args, **kwargs):
        raw_calls["n"] += 1
        return torch.zeros(2, 4)

    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    batch = {"query_gen_ids": ids, "query_gen_mask": mask}
    cfg = SimpleNamespace(data=SimpleNamespace(max_query_len=8),
                          query_encoder=SimpleNamespace(pooling="mean"))

    # A generator-style encoder: query_vector must use its pooled(), never the
    # module-level helper.
    seen = {"pooled": 0}

    def pooled(a, b):
        seen["pooled"] += 1
        return torch.ones(2, 4)

    generator_like = SimpleNamespace(last_pooled=None, pooled=pooled)
    model_module.pool_query_in_generator_space = counting
    try:
        vector = QuROModel.query_vector(
            SimpleNamespace(query_encoder=generator_like, cfg=cfg, lm=None), batch)
        check("A1/S go through the encoder, not the raw LM",
              seen["pooled"] == 1 and raw_calls["n"] == 0 and float(vector.mean()) == 1.0,
              f"pooled={seen['pooled']} raw={raw_calls['n']}")

        # A cached vector from this batch's forward wins, so C1 does not pay a
        # second 7B encode -- that was the point of last_pooled.
        cached = SimpleNamespace(last_pooled=torch.full((2, 4), 7.0), pooled=pooled)
        vector = QuROModel.query_vector(
            SimpleNamespace(query_encoder=cached, cfg=cfg, lm=None), batch)
        check("a vector already computed this step is reused, not recomputed",
              seen["pooled"] == 1 and float(vector.mean()) == 7.0)

        # Encoders that are not the generator have no adapter and no shared
        # dropout, so the raw call stays correct for them.
        plain = SimpleNamespace(last_pooled=None)
        QuROModel.query_vector(
            SimpleNamespace(query_encoder=plain, cfg=cfg, lm=None), batch)
        check("a non-generator encoder still falls back to the plain helper",
              raw_calls["n"] == 1)
    finally:
        model_module.pool_query_in_generator_space = original


def test_query_adapter_checkpoint():
    """A frozen query adapter must travel with the checkpoint and be verified.

    It is not trainable, so the generator-state filter drops it, and it cannot be
    rebuilt from the model path either: with ``generator_lora_init="random"`` the
    decoder adapter is reset *before* the copy is taken, and a local directory is
    not an immutable version.  Restoring a checkpoint against a different frozen
    adapter used to succeed silently -- a different system under the same name.
    """
    from src.model import GeneratorQueryEncoder, QuROModel

    class Stub:
        """Only the pieces ``_restore_query_adapter`` touches."""

        def __init__(self, adapter, weights, representation="fixed_adapter"):
            # Nested modules, so named_parameters() yields the dotted path the
            # adapter matching relies on; register_parameter rejects dots.
            leaf = torch.nn.Module()
            leaf.weight = torch.nn.Parameter(weights.clone())
            holder = torch.nn.Module()
            setattr(holder, adapter, leaf)
            self.lm = torch.nn.Module()
            self.lm.base = holder
            self.query_encoder = type("E", (), {})()
            self.query_encoder.query_adapter = adapter
            self.query_encoder.representation = representation
            self.query_encoder.query_adapter_hash = GeneratorQueryEncoder.adapter_hash(
                self.lm, adapter)

    torch.manual_seed(0)
    weights = torch.randn(4, 4)
    trained = Stub("quro_query_adapter", weights)
    payload = {
        "query_representation": "fixed_adapter",
        "query_adapter": {
            "name": "quro_query_adapter",
            "representation": "fixed_adapter",
            "hash": trained.query_encoder.query_adapter_hash,
            "state": {"base.quro_query_adapter.weight": weights.clone()},
        },
    }

    # A model rebuilt with a *different* frozen adapter must end up with the saved
    # one, not its own.
    rebuilt = Stub("quro_query_adapter", torch.randn(4, 4))
    different = rebuilt.query_encoder.query_adapter_hash
    QuROModel._restore_query_adapter(rebuilt, payload)
    check("loading restores the checkpoint's frozen query adapter",
          rebuilt.query_encoder.query_adapter_hash == payload["query_adapter"]["hash"]
          and different != payload["query_adapter"]["hash"])
    check("the restored adapter stays frozen",
          not any(p.requires_grad for n, p in rebuilt.lm.named_parameters()
                  if "quro_query_adapter" in n))

    # A corrupted payload must raise rather than load something else.
    corrupt = dict(payload)
    corrupt["query_adapter"] = dict(payload["query_adapter"], hash="deadbeefdeadbeef")
    try:
        QuROModel._restore_query_adapter(Stub("quro_query_adapter", weights), corrupt)
        check("a hash mismatch is rejected", False)
    except ValueError:
        check("a hash mismatch is rejected", True)

    # Strategy mismatch in both directions is a different system, not a warning.
    shared = Stub("quro_query_adapter", weights, representation="shared_current")
    shared.query_encoder.query_adapter = None
    try:
        QuROModel._restore_query_adapter(shared, payload)
        check("loading a fixed_adapter checkpoint into shared_current is rejected", False)
    except ValueError:
        check("loading a fixed_adapter checkpoint into shared_current is rejected", True)

    fixed = Stub("quro_query_adapter", weights)
    try:
        QuROModel._restore_query_adapter(
            fixed, {"query_representation": "shared_current"})
        check("loading a shared_current checkpoint into fixed_adapter is rejected", False)
    except ValueError:
        check("loading a shared_current checkpoint into fixed_adapter is rejected", True)


def test_query_slot_prompts(tokenizer, n_mem_tokens):
    """D4/D5 reserve separate positions for the compressed question.

    Both groups reuse the ``<MEM*>`` vocabulary -- adding a token would resize the
    embedding table and stop the PISCO comparison being like for like -- so they
    are told apart by order alone.  Writing the question into the evidence
    positions would produce a perfectly valid-looking prompt carrying the wrong
    content, which is why the counts are checked rather than assumed.
    """
    from src.prompt import PiscoPromptBuilder, assemble_inputs

    budget, query_tokens = 8, 6
    for mode in ("D4", "D5"):
        builder = PiscoPromptBuilder(tokenizer, n_mem_tokens, mode,
                                     query_tokens=query_tokens)
        prompt = builder.build("who directed the film", budget)
        check(f"{mode} reserves {budget} evidence slots",
              len(prompt.slot_positions) == budget, str(len(prompt.slot_positions)))
        check(f"{mode} reserves {query_tokens} question slots",
              len(prompt.query_slot_positions) == query_tokens,
              str(len(prompt.query_slot_positions)))
        check(f"{mode}: the two groups do not overlap",
              not (set(prompt.slot_positions) & set(prompt.query_slot_positions)))
        check(f"{mode}: evidence comes before the question",
              max(prompt.slot_positions) < min(prompt.query_slot_positions))

    # Length claims have to be tokenizer-independent.  "D4 is shorter than D0"
    # only holds when the question tokenizes to more than query_tokens, which is
    # true for Mistral (~23 tokens) and false for the word-level toy tokenizer --
    # so assert the structural identity instead: D4 is D1 with the question
    # replaced by exactly query_tokens embeddings.
    builders = {m: PiscoPromptBuilder(tokenizer, n_mem_tokens, m,
                                      query_tokens=query_tokens)
                for m in ("D0", "D1", "D4", "D5")}
    lengths = {m: len(b.build("who directed the film", budget).input_ids)
               for m, b in builders.items()}
    # The slot string carries a <SEP> per block, so the question costs
    # query_tokens + 1, not query_tokens.  Measure it rather than assume it.
    slot_cost = len(tokenizer(builders["D4"].slot_string(query_tokens),
                              add_special_tokens=False)["input_ids"])
    check("D4 is D1 plus exactly the question's slot string",
          lengths["D4"] == lengths["D1"] + slot_cost,
          f"{lengths} slot_cost={slot_cost}")
    check("the question's slots cost one separator beyond the slots themselves",
          slot_cost == query_tokens + 1, str(slot_cost))
    check("D5 drops the system prompt and scaffolding D4 keeps",
          lengths["D5"] < lengths["D4"], str(lengths))
    check("D5 carries only the slots plus BOS",
          lengths["D5"] <= budget + query_tokens + 4, str(lengths["D5"]))

    # The question embeddings must actually land in their own positions.
    hidden = 16
    embeddings = torch.nn.Embedding(max(len(tokenizer), 64), hidden)
    prompt = builders["D4"].build("who directed the film", budget)
    soft = torch.zeros(1, budget, hidden)
    question = torch.arange(1, query_tokens + 1, dtype=torch.float32)[None, :, None]
    question = question.expand(1, query_tokens, hidden).contiguous()
    packed = assemble_inputs(embeddings, [prompt], soft,
                             torch.ones(1, budget, dtype=torch.bool),
                             query_tokens=question)
    written = packed["inputs_embeds"][0, prompt.query_slot_positions, 0]
    check("the compressed question is written to its own slots",
          torch.allclose(written, torch.arange(1., query_tokens + 1.), atol=1e-5),
          str(written.tolist()))
    check("the evidence slots stay zero, not overwritten by the question",
          float(packed["inputs_embeds"][0, prompt.slot_positions].abs().max()) == 0.0)

    try:
        assemble_inputs(embeddings, [prompt], soft,
                        torch.ones(1, budget, dtype=torch.bool), query_tokens=None)
        check("a missing compressed question is rejected", False)
    except ValueError:
        check("a missing compressed question is rejected", True)


def test_distillation_loss():
    """The KL must behave like a divergence, and must not renormalise the top-k.

    The trap this guards: storing the teacher's top-k logits and softmaxing over
    those k gives a *different* distribution -- it deletes the tail and rescales
    what is left -- so a student that matched it exactly would still be wrong.
    The cache stores full-vocabulary probabilities plus the tail mass, and the
    divergence treats the tail as one aggregate bucket.
    """
    from src.distill import distillation_loss, teacher_probabilities

    torch.manual_seed(0)
    n, vocab, k, temperature = 6, 64, 8, 2.0
    teacher_logits = torch.randn(n, vocab) * 3
    index, probability, tail = teacher_probabilities(teacher_logits, k, temperature)

    check("stored teacher mass sums to one with the tail",
          torch.allclose(probability.sum(-1) + tail, torch.ones(n), atol=1e-5),
          f"max|diff|={float((probability.sum(-1) + tail - 1).abs().max()):.2e}")
    check("the tail is the mass outside the top-k, not zero",
          float(tail.min()) > 0 and float(tail.max()) < 1)

    # A student identical to the teacher must give (numerically) zero.
    same = distillation_loss(teacher_logits, index, probability, tail, temperature)
    check("KL(teacher || teacher) is zero", float(same) < 1e-5, f"{float(same):.2e}")

    # Any other student must give strictly more.
    other = distillation_loss(torch.randn(n, vocab) * 3, index, probability, tail,
                              temperature)
    check("a different student scores strictly higher", float(other) > float(same),
          f"{float(other):.4f} vs {float(same):.2e}")
    check("the divergence is never negative", float(other) >= 0)

    # Renormalising over the top-k is the mistake: a student fitted to *that*
    # distribution must not score zero against the correctly stored one.
    renormalised = probability / probability.sum(-1, keepdim=True)
    fake = torch.full((n, vocab), -30.0)
    fake.scatter_(-1, index, renormalised.clamp_min(1e-9).log() * temperature)
    wrong = distillation_loss(fake, index, probability, tail, temperature)
    check("a top-k-renormalised student does NOT match the stored teacher",
          float(wrong) > 1e-3, f"{float(wrong):.4f}")

    # Gradients must reach the student's logits, including through the tail term.
    student = (torch.randn(n, vocab) * 3).requires_grad_(True)
    distillation_loss(student, index, probability, tail, temperature).backward()
    check("the loss is differentiable w.r.t. the student",
          student.grad is not None and torch.isfinite(student.grad).all().item()
          and float(student.grad.abs().max()) > 0)

    # Misalignment must be an error, not a silently wrong objective.
    try:
        distillation_loss(student[:2], index, probability, tail, temperature)
        check("a row-count mismatch is rejected", False)
    except ValueError:
        check("a row-count mismatch is rejected", True)


def test_answer_positions():
    """Answer positions are read with the causal shift and indexed relatively.

    Teacher and student prompts differ in length (80 soft tokens against 8), so
    absolute positions do not correspond; only the answer-relative index does.
    """
    from src.model import QuROModel

    # Two rows, different prompt lengths, same two-token answer.
    labels = torch.tensor([
        [-100, -100, -100, 11, 12, -100],
        [-100, 11, 12, -100, -100, -100],
    ])
    rows, cols, order = QuROModel.answer_positions(labels)
    check("one position per answer token", rows.numel() == 4)
    check("answer-relative index restarts per row",
          order.tolist() == [0, 1, 0, 1], str(order.tolist()))
    # logits[t] predicts labels[t+1], so the columns are one before the labels.
    check("the causal shift is applied", cols.tolist() == [2, 3, 0, 1],
          str(cols.tolist()))
    check("the predicted tokens are the answer tokens",
          labels[:, 1:][rows, cols].tolist() == [11, 12, 11, 12])


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
        test_cosine_prior_short_rows(m, h, h)
        test_output_modes(m, h, h)
        test_distillation_loss()
        test_answer_positions()
        test_arm_labels()
        test_query_representation()
        test_query_vector_routing()
        test_query_adapter_checkpoint()
        test_query_path_separation(cache_dir, train_path, m, h)
        test_baselines(m, h)
        test_cache(cache_dir, doc_ids, m, h)

        from src.toy import ToyTokenizer
        toy_tok = ToyTokenizer.build_from_texts(
            ["who designed the building and in which year", "who directed the film"])
        test_prompt(toy_tok, 8)
        test_query_slot_prompts(toy_tok, 8)

        test_end_to_end(cache_dir, train_path, m, h)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed:", ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
