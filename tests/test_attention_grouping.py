"""Contract tests for the full-compression infeasibility diagnostics -- CPU only.

Every number in ``docs/FULL_COMPRESSION_INFEASIBILITY_RESULTS.md`` rests on two
things being exactly right, and neither is visible in a finished plot:

* the **position manifest** -- which decoder positions are "the memory" and which
  are "the instruction".  A boundary off by one token moves mass between the two
  groups whose ratio is the headline number;
* the **recording attention** -- ``src/infeasibility.capture_attention`` replaces
  ``eager_attention_forward`` with a reimplementation, so it has to be checked
  against the original and against attention worked out by hand.

Run with ``python tests/test_attention_grouping.py``.
"""

from __future__ import annotations

import math
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import infeasibility as inf
from src import paths
from src.prompt import PiscoPromptBuilder

PASS, FAIL = [], []


def check(name, condition, detail=""):
    (PASS if condition else FAIL).append(name)
    print(f"{'PASS' if condition else 'FAIL'} {name} {detail}")


# ----------------------------------------------------------------------------
# A PISCO-shaped tokenizer, loaded from the local Mistral copy the same way
# modelling_pisco.create_decoder_tokenizer does.
# ----------------------------------------------------------------------------
def build_tokenizer(n_mem_tokens: int = 8):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(paths.MISTRAL_PATH, use_fast=True,
                                              padding_side="left")
    mem_tokens = [f"<MEM{i}>" for i in range(n_mem_tokens)]
    tokenizer.add_special_tokens(
        {"additional_special_tokens": mem_tokens + ["<AE>", "<ENC>", "<SEP>"]})
    tokenizer.mem_tokens = mem_tokens
    tokenizer.mem_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in mem_tokens]
    tokenizer.ae_token, tokenizer.enc_token = "<AE>", "<ENC>"
    tokenizer.sep_token = "<SEP>"
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("<SEP>")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.bos_token_id
    return tokenizer


DOCUMENT = ("Title: Olaf M. Hustvedt\nContent: A grandson, Frederick Hauck (b. 1941), "
            "became a U.S. Navy officer, fighter pilot, and astronaut in the "
            "National Aeronautics and Space Administrations Space Shuttle Program.")
QUERY = "What is Olaf M. Hustvedt's grandson's occupation?"
INSTRUCTION = inf.CONFLICT_TEMPLATE.format(nonce="SDJKLGHFLKJA")


# ----------------------------------------------------------------------------
# 1. Position manifest
# ----------------------------------------------------------------------------
def test_manifest(tokenizer):
    builder = PiscoPromptBuilder(tokenizer, n_mem_tokens=8)
    budget = 16
    slots = inf.slot_string(tokenizer, budget, 8)

    # The rendering must *be* the repository's PISCO prompt, not a lookalike:
    # D0 with memory slots and RG with raw text, token for token.
    compressed = inf.token_groups(tokenizer, inf.render(tokenizer, slots, query=QUERY))
    reference = builder.build(QUERY, budget)
    check("D0 rendering is token-identical to PiscoPromptBuilder",
          compressed.input_ids == reference.input_ids,
          f"{len(compressed.input_ids)} vs {len(reference.input_ids)} tokens")
    check("memory slot positions match PiscoPromptBuilder",
          inf.memory_positions(tokenizer, compressed) == reference.slot_positions)

    raw_builder = PiscoPromptBuilder(tokenizer, n_mem_tokens=8, mode="RG")
    raw = inf.token_groups(tokenizer, inf.render(tokenizer, DOCUMENT, query=QUERY))
    check("RG rendering is token-identical to PiscoPromptBuilder",
          raw.input_ids == raw_builder.build(QUERY, budget, [DOCUMENT]).input_ids)

    # Groups tile the sequence: no overlap, no gap, exact length.
    groups = inf.token_groups(
        tokenizer, inf.render(tokenizer, slots, query=QUERY, instruction=" " + INSTRUCTION))
    groups.check()
    sizes = groups.sizes()
    check("group sizes sum to the token count",
          sum(sizes.values()) == len(groups.input_ids), str(sizes))
    check("memory group holds exactly the <MEM*> tokens",
          sizes["document"] == budget + budget // 8,      # slots plus one <SEP> per block
          f"document={sizes['document']} budget={budget}")
    check("instruction group is non-empty", sizes["instruction"] > 0)
    check("query group is non-empty", sizes["query"] > 0)

    # Hand-checkable: decoding each group's tokens must give back its own text.
    start, end = groups.spans["instruction"]
    decoded = tokenizer.decode(groups.input_ids[start:end])
    check("instruction group decodes to the instruction", INSTRUCTION in decoded,
          repr(decoded[:60]))
    start, end = groups.spans["query"]
    check("query group decodes to the query",
          QUERY in tokenizer.decode(groups.input_ids[start:end]))

    # Absent groups still have to be located, or the manifest cannot say where a
    # missing query would have gone.
    no_query = inf.token_groups(
        tokenizer, inf.render(tokenizer, slots, query="", instruction=INSTRUCTION))
    a, b = no_query.spans["query"]
    check("empty query group is zero-length but positioned", a == b and a > 0)

    no_memory = inf.token_groups(
        tokenizer, inf.render(tokenizer, None, query=QUERY, instruction=INSTRUCTION))
    check("no-memory prompt has no document group",
          "document" not in no_memory.spans and "document_delimiter" not in no_memory.spans)
    no_memory.check()

    # Without a system prompt the template emits "<s> [INST] Background:", and
    # SentencePiece merges the space into the next token, so the prefix boundary
    # lands mid-token.  That must still produce a valid partition.
    for style in inf.PROMPT_STYLES:
        for system in (inf.SYSTEM_PROMPT, None):
            variant = inf.token_groups(
                tokenizer, inf.render(tokenizer, slots, query=QUERY,
                                      instruction=" " + INSTRUCTION,
                                      system_prompt=system, style=style))
            variant.check()
            tag = f"{style}/{'system' if system else 'no-system'}"
            check(f"[{tag}] groups still tile the sequence",
                  sum(variant.sizes().values()) == len(variant.input_ids))
            check(f"[{tag}] memory slots are all inside the document group",
                  len(inf.memory_positions(tokenizer, variant)) == budget)
            # token_groups refuses a boundary that would split real content, so
            # reaching here means every crossing was whitespace; record how many.
            check(f"[{tag}] at most one token crosses a boundary",
                  len(variant.straddling_tokens) <= 1,
                  f"straddling={variant.straddling_tokens}")
            # The straddling token spells "Background", and it belongs with the
            # delimiter it names rather than with the space in front of it.
            start, end = variant.spans["document_delimiter"]
            check(f"[{tag}] the Background marker is in document_delimiter",
                  "Background" in tokenizer.decode(variant.input_ids[start:end]),
                  repr(tokenizer.decode(variant.input_ids[start:end])))
            head = tokenizer.decode(variant.input_ids[slice(*variant.spans["prefix"])])
            check(f"[{tag}] prefix does not swallow the Background marker",
                  "Background" not in head, repr(head[-24:]))

    # A boundary with real content on both sides is still refused.
    raised = False
    try:
        inf.token_groups(tokenizer, inf.RenderedPrompt(
            text="alphabeta", char_spans={"prefix": (0, 3), "query_delimiter": (3, 4),
                                          "query": (4, 5), "instruction": (5, 8),
                                          "answer_prefix": (8, 9)}))
    except ValueError:
        raised = True
    check("a boundary that would split real content is refused", raised)

    # Teacher-forced targets extend the manifest without disturbing it.
    targets = tokenizer(" SDJKLGHFLKJA", add_special_tokens=False)["input_ids"]
    extended = groups.with_targets(targets)
    a, b = extended.spans["output_history"]
    check("target slice sits immediately after the prompt",
          a == len(groups.input_ids) and b - a == len(targets))
    check("target slice holds the target tokens",
          extended.input_ids[a:b] == list(targets))
    index = extended.index_tensor(len(extended.input_ids))
    check("every position carries a group id", bool((index >= 0).all()))
    check("target positions are tagged output_history",
          bool((index[a:b] == inf.GROUP_INDEX["output_history"]).all()))

    # Spans that do not tile the text must be an error, not a silent rounding.
    try:
        inf.RenderedPrompt(
            text="abcdefg", char_spans={"prefix": (0, 2), "query_delimiter": (2, 3),
                                        "query": (3, 4), "instruction": (4, 5),
                                        "answer_prefix": (5, 6)}).check()
        raised = False
    except ValueError:
        raised = True
    check("a span set that leaves text uncovered is rejected", raised)


# ----------------------------------------------------------------------------
# 2. Grouped mass / density, with an explicit hand computation
# ----------------------------------------------------------------------------
def test_grouping_math():
    # Six positions in three groups; two target rows.  Small enough to check by eye.
    group_index = torch.tensor([0, 0, 2, 2, 2, 5])   # prefix, prefix, doc x3, instruction
    targets = torch.tensor([4, 5])
    collector = inf._GroupCollector(group_index, targets, n_layers=1)

    heads, total = 2, 6
    probs = torch.zeros(heads, total, total)
    for t in range(total):
        probs[:, t, : t + 1] = 1.0 / (t + 1)         # uniform over the causal prefix
    pre = torch.arange(heads * total * total, dtype=torch.float32).reshape(heads, total, total)
    key = torch.ones(heads, total, 4)
    value = torch.ones(heads, total, 4)
    key[:, 2:5] *= 3.0                                # a "polarised" document norm
    collector.record(0, pre, probs, key, value, scaling=1.0)
    got = collector.layers[0]

    # Target t=4 sees positions 0..4 uniformly at 1/5: prefix 2/5, document 3/5,
    # instruction 0 (position 5 is in the future).
    mass = got["mass"]
    check("mass on the prefix at t=4", math.isclose(float(mass[0, 0, 0]), 2 / 5, abs_tol=1e-6),
          f"{float(mass[0, 0, 0]):.4f}")
    check("mass on the document at t=4", math.isclose(float(mass[0, 0, 2]), 3 / 5, abs_tol=1e-6))
    check("mass on the not-yet-visible instruction at t=4",
          math.isclose(float(mass[0, 0, 5]), 0.0, abs_tol=1e-6))
    check("mass sums to one over all groups at t=4",
          math.isclose(float(mass[0, 0].sum()), 1.0, abs_tol=1e-5))
    # Target t=5 sees all six: prefix 2/6, document 3/6, instruction 1/6.
    check("mass on the instruction at t=5",
          math.isclose(float(mass[0, 1, 5]), 1 / 6, abs_tol=1e-6))

    # Density divides by the *causally visible* group size, so the instruction is
    # NaN at t=4 rather than zero: it does not yet exist for that target.
    collected = collector.finish(None)
    density = collected.density
    check("group size counts only visible tokens",
          collected.group_size[0, 5] == 0 and collected.group_size[1, 5] == 1,
          str(collected.group_size.tolist()))
    check("density on the document at t=5",
          math.isclose(float(density[0, 0, 1, 2]), (3 / 6) / 3, abs_tol=1e-6))
    check("density of an invisible group is NaN, not zero",
          bool(np.isnan(density[0, 0, 0, 5])))

    # Pre-softmax logits are averaged over visible positions only.
    expected = float(pre[0, 4, 2:5].mean())
    check("qk_mean over the document at t=4",
          math.isclose(float(got["qk_mean"][0, 0, 2]), expected, rel_tol=1e-5),
          f"{float(got['qk_mean'][0, 0, 2])} vs {expected}")
    check("qk_max over the document at t=4",
          math.isclose(float(got["qk_max"][0, 0, 2]), float(pre[0, 4, 2:5].max()), rel_tol=1e-5))
    check("qk over an invisible group is NaN", bool(np.isnan(got["qk_mean"][0, 0, 5])))

    # K norms are per-group means of ||k_j||; the document was scaled by 3.
    check("grouped key norm separates the document",
          math.isclose(float(got["k_norm"][0, 2]) / float(got["k_norm"][0, 0]), 3.0, rel_tol=1e-4),
          f"{float(got['k_norm'][0, 2])} vs {float(got['k_norm'][0, 0])}")

    # Value contribution: ||sum_j A_tj v_j|| with v all-ones in 4 dimensions.
    expected = (3 / 6) * 2.0                          # mass 1/2 times ||(1,1,1,1)||=2
    check("value-weighted contribution of the document at t=5",
          math.isclose(float(got["v_contrib"][0, 1, 2]), expected, rel_tol=1e-4),
          f"{float(got['v_contrib'][0, 1, 2])} vs {expected}")

    # Entropy of a uniform distribution over t+1 positions is log(t+1).
    check("entropy of the uniform row at t=5",
          math.isclose(float(got["entropy"][0, 1]), math.log(6), rel_tol=1e-4))
    check("top attended group is reported", int(got["top_group"][0, 1]) in (0, 2, 5))

    # Per-layer contextualisation: a big residual with a small update barely
    # moves, which is the whole "frozen outlier" question.
    big, small = 100.0, 0.1
    h0 = torch.zeros(1, total, 4)
    h0[0, :2] = small / 2.0            # prefix: residual of norm 0.1
    h0[0, 2:5] = big / 2.0             # document: residual of norm 100
    h0[0, 5] = small / 2.0
    # The same unit update everywhere, orthogonal to the residual so the rotation
    # it causes is a pure function of the magnitude ratio.
    update = torch.tensor([0.5, -0.5, 0.5, -0.5]).expand(1, total, 4).contiguous()
    collected = collector.finish([h0, h0 + update])
    prefix = collected.hidden_update[0, 0]
    document = collected.hidden_update[0, 2]
    check("identical updates move a small residual far more than a large one",
          prefix > 20 * document, f"prefix {prefix:.4f} vs document {document:.6f}")
    check("the large residual barely rotates",
          collected.hidden_rotation[0, 2] > 0.9999,
          f"cos = {collected.hidden_rotation[0, 2]:.6f}")
    check("the small residual rotates a lot",
          collected.hidden_rotation[0, 0] < 0.99,
          f"cos = {collected.hidden_rotation[0, 0]:.6f}")


# ----------------------------------------------------------------------------
# 3. The recording attention must be the original
# ----------------------------------------------------------------------------
def test_patch_fidelity():
    from transformers.models.mistral import modeling_mistral as mistral

    torch.manual_seed(0)
    heads, kv_heads, total, dim = 4, 2, 9, 8

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.num_key_value_groups = heads // kv_heads
            self.layer_idx = 0

    module = Stub().eval()
    query = torch.randn(1, heads, total, dim)
    key = torch.randn(1, kv_heads, total, dim)
    value = torch.randn(1, kv_heads, total, dim)
    causal = torch.full((1, 1, total, total), torch.finfo(torch.float32).min)
    causal = causal.triu(1)
    scaling = 1.0 / math.sqrt(dim)

    reference_out, reference_weights = mistral.eager_attention_forward(
        module, query, key, value, causal, scaling)

    group_index = torch.zeros(total, dtype=torch.long)
    group_index[3:6] = inf.GROUP_INDEX["document"]
    group_index[6:] = inf.GROUP_INDEX["instruction"]
    collector = inf._GroupCollector(group_index, torch.tensor([total - 1]), n_layers=1)
    with inf.capture_attention(collector):
        patched_out, patched_weights = mistral.eager_attention_forward(
            module, query, key, value, causal, scaling)

    check("patched attention output matches the original",
          torch.allclose(reference_out, patched_out, atol=1e-6),
          f"max diff {float((reference_out - patched_out).abs().max()):.2e}")
    check("patched attention weights match the original",
          torch.allclose(reference_weights, patched_weights, atol=1e-6))
    check("the patch is removed on exit",
          mistral.eager_attention_forward.__name__ == "eager_attention_forward")

    # The recorded mass must equal the weights summed over each group directly.
    recorded = collector.layers[0]["mass"]
    direct = reference_weights[0, :, -1, :]
    for name in ("document", "instruction"):
        g = inf.GROUP_INDEX[name]
        expected = direct[:, group_index == g].sum(-1)
        check(f"recorded mass on {name} equals the summed weights",
              torch.allclose(torch.tensor(recorded[:, 0, g]), expected, atol=1e-5),
              f"{recorded[:, 0, g]} vs {expected.tolist()}")

    # Batches larger than one would let left padding shift the manifest silently.
    collector2 = inf._GroupCollector(group_index, torch.tensor([total - 1]), n_layers=1)
    raised = False
    try:
        with inf.capture_attention(collector2):
            mistral.eager_attention_forward(
                module, query.repeat(2, 1, 1, 1), key.repeat(2, 1, 1, 1),
                value.repeat(2, 1, 1, 1), causal, scaling)
    except ValueError:
        raised = True
    check("a batched diagnostic forward is refused", raised)


# ----------------------------------------------------------------------------
# 4. Behavioural metrics
# ----------------------------------------------------------------------------
def test_metrics():
    nonce = "SDJKLGHFLKJA"
    check("exact nonce output scores 1",
          inf.score_conflict(f'"{nonce}"', nonce, DOCUMENT)["exact"] == 1.0)
    chatty = inf.score_conflict(f"Sure: {nonce}", nonce, DOCUMENT)
    check("a compliant but chatty answer counts as leading",
          chatty["exact"] == 1.0 and chatty["leading"] == 1.0, str(chatty["nonce_share"]))
    recited = inf.score_conflict(DOCUMENT, nonce, DOCUMENT)
    check("reciting the document scores zero on the nonce", recited["leading"] == 0.0)
    check("reciting the document is caught by the copied span",
          recited["longest_copied_span"] > 20, str(recited["longest_copied_span"]))

    # The failure mode that made the first smoke run look like 100% compliance.
    refusal = inf.score_conflict(
        f'I\'m sorry, but the provided document does not contain the string "{nonce}". '
        f"The document only describes the Liber Paradisus, a law text from Bologna.",
        nonce, DOCUMENT)
    check("a refusal that quotes the nonce is not instruction following",
          refusal["exact"] == 0.0 and refusal["leading"] == 0.0)
    check("the same refusal is still counted by the mentions upper bound",
          refusal["mentions"] == 1.0)
    check("nonce_share separates a refusal from compliance",
          refusal["nonce_share"] < 0.1 < chatty["nonce_share"],
          f"{refusal['nonce_share']:.3f} vs {chatty['nonce_share']:.3f}")

    perfect = inf.score_reconstruction(DOCUMENT, DOCUMENT)
    check("verbatim reconstruction scores 1", perfect["exact"] == 1.0
          and math.isclose(perfect["rouge_l"], 1.0, abs_tol=1e-9))
    half = inf.score_reconstruction(DOCUMENT.split("Content:")[0], DOCUMENT)
    check("partial reconstruction scores between 0 and 1",
          0.0 < half["rouge_l"] < 1.0, f"{half['rouge_l']:.3f}")

    boot = inf.paired_bootstrap([1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0], seed=1)
    check("paired bootstrap recovers a constant difference",
          math.isclose(boot["delta"], 1.0, abs_tol=1e-9) and boot["lo"] == boot["hi"] == 1.0)
    noisy = inf.paired_bootstrap([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], seed=1)
    check("a bootstrap interval spanning zero is reported as such",
          noisy["lo"] < 0 < noisy["hi"], f"[{noisy['lo']:.2f}, {noisy['hi']:.2f}]")


def test_transforms():
    """Each §11 edit must remove exactly the property it claims to, and no other."""
    torch.manual_seed(0)
    latents = torch.randn(8, 16) * torch.linspace(1.0, 4.0, 8)[:, None]
    norms = latents.norm(dim=-1)

    same = inf.apply_transform(latents, "none")
    check("none leaves the latents untouched", torch.equal(same, latents))

    half = inf.apply_transform(latents, "scale:0.5")
    check("scale halves every latent", torch.allclose(half, latents * 0.5))

    generator = torch.Generator().manual_seed(1)
    random = inf.apply_transform(latents, "norm_matched_random", generator)
    check("norm_matched_random preserves the per-slot norms",
          torch.allclose(random.norm(dim=-1), norms, rtol=1e-3),
          f"{random.norm(dim=-1)[:3].tolist()} vs {norms[:3].tolist()}")
    cosine = torch.nn.functional.cosine_similarity(random, latents, dim=-1)
    check("norm_matched_random discards the directions",
          bool(cosine.abs().max() < 0.7), f"max |cos| = {float(cosine.abs().max()):.3f}")

    generator = torch.Generator().manual_seed(2)
    shuffled = inf.apply_transform(latents, "shuffle", generator)
    check("shuffle keeps the multiset of latents",
          torch.allclose(shuffled.norm(dim=-1).sort().values, norms.sort().values))
    check("shuffle actually reorders", not torch.equal(shuffled, latents))

    averaged = inf.apply_transform(latents, "mean")
    check("mean makes every slot identical",
          torch.allclose(averaged, averaged[0].expand_as(averaged)))
    check("mean keeps the row's centre",
          torch.allclose(averaged.mean(0), latents.mean(0), atol=1e-5))

    raised = False
    try:
        inf.Condition("x", "raw", "conflict", "scale:0.5")
    except ValueError:
        raised = True
    check("a transform on a non-memory condition is refused", raised)

    # Corpus-level transforms: a fixed mean and basis, checked by construction.
    torch.manual_seed(3)
    mu = torch.randn(16) * 4.0
    basis = torch.linalg.qr(torch.randn(16, 4))[0].T.contiguous()      # (4, 16)
    stats = inf.LatentStatistics(mean=mu, basis=basis,
                                 singular_values=torch.arange(4, 0, -1).float(),
                                 n_documents=10, n_vectors=80, seed=0)

    gmean = inf.apply_transform(latents, "global_mean", stats=stats)
    check("global_mean puts the corpus mean in every slot",
          torch.allclose(gmean, mu[None].expand_as(latents)))

    dec = inf.apply_transform(latents, "decenter", stats=stats)
    check("decenter subtracts the corpus mean", torch.allclose(dec, latents - mu[None]))
    check("decenter changes the norm", not torch.allclose(dec.norm(dim=-1), norms))

    decn = inf.apply_transform(latents, "decenter_renorm", stats=stats)
    check("decenter_renorm holds the per-slot norm fixed",
          torch.allclose(decn.norm(dim=-1), norms, rtol=1e-4),
          f"{decn.norm(dim=-1)[:2].tolist()} vs {norms[:2].tolist()}")
    check("decenter_renorm keeps the decentred direction",
          torch.allclose(torch.nn.functional.cosine_similarity(decn, dec, dim=-1),
                         torch.ones(latents.size(0)), atol=1e-4))

    proj = inf.apply_transform(latents, "deproject:4", stats=stats)
    residual = proj @ basis.T
    check("deproject removes the basis directions",
          bool(residual.abs().max() < 1e-4), f"max |Z.v| = {float(residual.abs().max()):.2e}")
    check("deproject holds the per-slot norm fixed",
          torch.allclose(proj.norm(dim=-1), norms, rtol=1e-4))
    partial = inf.apply_transform(latents, "deproject:1", stats=stats)
    check("deproject:1 removes only the first direction",
          bool((partial @ basis[:1].T).abs().max() < 1e-4)
          and bool((partial @ basis[1:].T).abs().max() > 1e-3))

    raised = False
    try:
        inf.apply_transform(latents, "decenter")
    except ValueError:
        raised = True
    check("a corpus transform without corpus statistics is refused", raised)

    names = {c.name for c in inf.subspace_conditions()}
    check("the subspace sweep measures both tasks",
          all(n.replace("/conflict", "/reconstruct") in names
              for n in names if n.startswith("memory@") and n.endswith("/conflict")))
    check("the subspace sweep includes the global-mean complement",
          "memory@gmean/conflict" in names and "memory@rmean/conflict" in names)

    names = {c.name for c in inf.intervention_conditions()}
    check("every intervention is measured on both tasks",
          all(n.replace("/conflict", "/reconstruct") in names
              for n in names if n.startswith("memory@") and n.endswith("/conflict")),
          str(sorted(names)))
    check("the intervention sweep carries its own observational anchors",
          {"memory/conflict", "raw/conflict", "zero/conflict", "none/conflict"} <= names)


def test_nonce(tokenizer):
    rng = random.Random(0)
    for length in (8, 12, 16):
        payload = inf.make_nonce(rng, tokenizer, length)
        check(f"nonce of {length} chars round-trips through the tokenizer",
              tokenizer.decode(payload["nonce_token_ids"], skip_special_tokens=True)
              == payload["nonce"], payload["nonce"])
    first = inf.make_nonce(random.Random(1), tokenizer)["nonce"]
    second = inf.make_nonce(random.Random(2), tokenizer)["nonce"]
    check("nonces differ between examples", first != second, f"{first} / {second}")


def main():
    if not os.path.exists(paths.MISTRAL_PATH):
        print(f"SKIP tokenizer tests: {paths.MISTRAL_PATH} is absent")
        tokenizer = None
    else:
        tokenizer = build_tokenizer()

    if tokenizer is not None:
        test_manifest(tokenizer)
        test_nonce(tokenizer)
    test_grouping_math()
    test_patch_fidelity()
    test_transforms()
    test_metrics()

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("failed: " + ", ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
