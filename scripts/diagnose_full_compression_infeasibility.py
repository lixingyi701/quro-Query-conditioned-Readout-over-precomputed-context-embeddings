"""Does SeleCom's "full compression is infeasible" claim reproduce on PISCO?

SeleCom §3.2 reports that a generator reading full-compression embeddings keeps
reciting the document when the instruction says to ignore it, and that its
attention stays pinned on the compressed positions.  This script asks the same
question of *our* PISCO baseline -- the ``P`` arm, loaded through the repository's
own generator stack, prompt scaffolding and latent cache, with nothing trained --
and it collects the evidence needed to tell the three candidate explanations
apart rather than only the picture.

Two levels, as required by ``docs/SELECOM_FULL_COMPRESSION_INFEASIBILITY_INSTRUCTION.md`` §5:

``--level A``
    Single short documents, hand-checkable.  The minimal reproduction of
    SeleCom's Figure 2 with its own two prompts.

``--level B``
    HotpotQA dev at the repository's fixed K=10, where the project's actual
    numbers live.  Single- and multi-document rows are recorded separately,
    because ``K * m`` memory positions against a ~20-token instruction would
    "confirm" SeleCom on token count alone.

Both levels run the qualification gate first (§4): correct memory against
mismatched, absent and zeroed memory.  If PISCO memory does not measurably carry
the document, "it suppressed the instruction" is not a claim the data supports,
and the run says so instead of proceeding to heatmaps.

Everything is written to a fresh ``results/full_compression_infeasibility/<run_id>/``
directory; nothing existing is touched.

    python scripts/diagnose_full_compression_infeasibility.py --level A --rows 40
    python scripts/diagnose_full_compression_infeasibility.py --level B --rows 200
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import apply_arm, get_config
from src import infeasibility as inf
from src import paths
from src.cache import LatentCache
from src.data import read_jsonl
from src.model import build_model
from src.prompt import BuiltPrompt, assemble_inputs


# --------------------------------------------------------------------------------------
# Run provenance
# --------------------------------------------------------------------------------------
def git_state() -> Dict[str, object]:
    def run(*args):
        try:
            return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return ""
    return {"commit": run("git", "rev-parse", "HEAD"),
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("git", "status", "--porcelain"))}


def environment() -> Dict[str, object]:
    import transformers

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda, "gpus": gpus}


# --------------------------------------------------------------------------------------
# Sample construction
# --------------------------------------------------------------------------------------
def clip_to_fed_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """The text PISCO actually consumed.

    The compressor hard-truncates at ``doc_max_length``; showing the raw-text
    control the untruncated passage would give it evidence the compressed arm
    never had, and the comparison would stop being about compression.
    """
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def load_corpus(path: str) -> Dict[str, str]:
    return {str(r["doc_id"]): str(r.get("text", r.get("document", "")))
            for r in read_jsonl(path)}


def level_a_samples(tokenizer, corpus: Dict[str, str], cache: LatentCache,
                    rows: int, rng: random.Random,
                    min_tokens: int = 32, max_tokens: int = 96) -> List[Dict]:
    """Short, single, uniquely identifiable passages (§5 Level A).

    Sampled from the same HotpotQA corpus the cache was built over, so the
    latents are the ones the real system serves rather than a fresh compression
    of a hand-written toy document.
    """
    candidates = [d for d in corpus if d in cache]
    rng.shuffle(candidates)
    samples: List[Dict] = []
    for doc_id in candidates:
        if len(samples) >= rows:
            break
        text = corpus[doc_id]
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if not (min_tokens <= n_tokens <= max_tokens):
            continue
        # A passage with no name, number or date in it cannot be told apart from
        # a plausible hallucination, so reconstruction would not be scorable.
        if not any(c.isdigit() for c in text) and text.count(" ") < 20:
            continue
        samples.append({"sample_id": f"A-{len(samples):04d}", "doc_ids": [doc_id],
                        "document_tokens": n_tokens, "query": "", "answers": []})
    if len(samples) < rows:
        raise SystemExit(f"only {len(samples)} corpus documents fall in "
                         f"[{min_tokens}, {max_tokens}] tokens and are cached")
    # A deterministic neighbour supplies the mismatched memory, so the control is
    # reproducible from the manifest rather than from a live RNG.
    for i, sample in enumerate(samples):
        sample["mismatch_doc_ids"] = samples[(i + 1) % len(samples)]["doc_ids"]
    return samples


def level_b_samples(queries_path: str, cache: LatentCache, rows: int,
                    max_docs: int, rng: random.Random) -> List[Dict]:
    """HotpotQA dev rows at the repository's fixed K, untouched test split."""
    all_rows = read_jsonl(queries_path)
    index = list(range(len(all_rows)))
    rng.shuffle(index)
    samples: List[Dict] = []
    for i in index:
        if len(samples) >= rows:
            break
        row = all_rows[i]
        doc_ids = [str(x) for x in row["retrieved_doc_ids"]][:max_docs]
        if not doc_ids or any(d not in cache for d in doc_ids):
            continue
        samples.append({
            "sample_id": f"B-{len(samples):04d}", "row_id": str(row["id"]),
            "doc_ids": doc_ids, "query": str(row["query"]),
            "answers": [str(a) for a in row.get("answers", [])],
            "gold_ranks": row.get("gold_ranks", []),
            "hop_type": row.get("hop_type"), "n_docs": len(doc_ids)})
    if len(samples) < rows:
        raise SystemExit(f"only {len(samples)} of {len(all_rows)} dev rows are fully cached")
    for i, sample in enumerate(samples):
        sample["mismatch_doc_ids"] = samples[(i + 1) % len(samples)]["doc_ids"]
    return samples


# --------------------------------------------------------------------------------------
# Prompt / input assembly
# --------------------------------------------------------------------------------------
class PromptFactory:
    """Turns (sample, condition) into embeddings plus an auditable group manifest."""

    def __init__(self, tokenizer, lm, cache: LatentCache, corpus: Dict[str, str],
                 n_mem_tokens: int, doc_max_tokens: int,
                 style: str = "pisco_question",
                 system_prompt: Optional[str] = None):
        self.tok = tokenizer
        self.lm = lm
        self.cache = cache
        self.corpus = corpus
        self.n_mem_tokens = n_mem_tokens
        self.doc_max_tokens = doc_max_tokens
        self.hidden = int(lm.config.hidden_size)
        self.pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        self.style = style
        self.system_prompt = system_prompt

    def document_text(self, doc_ids: Sequence[str]) -> str:
        """The raw-text rendering, clipped exactly as the compressor's input was."""
        return "\n\n".join(clip_to_fed_tokens(self.tok, self.corpus[d], self.doc_max_tokens)
                           for d in doc_ids)

    def latents(self, doc_ids: Sequence[str], device, dtype) -> torch.Tensor:
        stacked, _, _ = self.cache.get_many([list(doc_ids)])
        return stacked[0].reshape(-1, self.hidden).to(device=device, dtype=dtype)

    def build(self, sample: Dict, condition: inf.Condition, instruction: str,
              query: str, device) -> Dict:
        dtype = self.lm.get_input_embeddings().weight.dtype
        kind = condition.document_kind
        doc_ids = sample["doc_ids"]
        n_latents = len(doc_ids) * self.cache.metadata.latent_size
        # Leakage has to be scored against the document the model was actually
        # shown: under the mismatch control that is the neighbour's passage, and
        # scoring it against the correct one would record a recital as clean.
        shown_ids = sample["mismatch_doc_ids"] if kind == "mismatch" else doc_ids

        if kind == "raw":
            document = self.document_text(doc_ids)
            soft = None
        elif kind == "none":
            document = None
            soft = None
        else:
            document = inf.slot_string(self.tok, n_latents, self.n_mem_tokens)
            if kind == "memory":
                soft = self.latents(doc_ids, device, dtype)
            elif kind == "mismatch":
                soft = self.latents(sample["mismatch_doc_ids"], device, dtype)
                # Mismatched evidence must not also change the *length* of the
                # memory, or a behaviour change would be attributable to either.
                soft = self._match_length(soft, n_latents)
            else:  # zero
                soft = torch.zeros(n_latents, self.hidden, device=device, dtype=dtype)

        rendered = inf.render(self.tok, document, query=query, instruction=instruction,
                              system_prompt=self.system_prompt, style=self.style)
        groups = inf.token_groups(self.tok, rendered)
        mem_positions = inf.memory_positions(self.tok, groups)
        if soft is not None and len(mem_positions) != soft.size(0):
            raise ValueError(f"{len(mem_positions)} memory slots for {soft.size(0)} latents")
        return {"groups": groups, "soft": soft, "mem_positions": mem_positions,
                "shown_document_text": "" if kind == "none" else self.document_text(shown_ids)}

    @staticmethod
    def _match_length(soft: torch.Tensor, n_latents: int) -> torch.Tensor:
        if soft.size(0) == n_latents:
            return soft
        if soft.size(0) > n_latents:
            return soft[:n_latents]
        repeats = -(-n_latents // soft.size(0))
        return soft.repeat(repeats, 1)[:n_latents]

    def assemble(self, built: Dict, target_ids: Optional[Sequence[int]], device,
                 pad_side: str = "right"):
        """Embeddings, via the same ``assemble_inputs`` the P baseline uses."""
        groups = built["groups"]
        prompt = BuiltPrompt(input_ids=list(groups.input_ids),
                             slot_positions=list(built["mem_positions"]))
        soft = built["soft"]
        if soft is None:
            soft = torch.zeros(1, self.hidden, device=device,
                               dtype=self.lm.get_input_embeddings().weight.dtype)
            mask = torch.zeros(1, 1, dtype=torch.bool, device=device)
            soft = soft[None]
        else:
            soft = soft[None]
            mask = torch.ones(1, soft.size(1), dtype=torch.bool, device=device)
        return assemble_inputs(
            self.lm.get_input_embeddings(), [prompt], soft, mask,
            target_ids=None if target_ids is None else [list(target_ids)],
            pad_token_id=self.pad_id, pad_side=pad_side)


# --------------------------------------------------------------------------------------
# Task definition: instruction text and teacher-forced target
# --------------------------------------------------------------------------------------
def task_texts(sample: Dict, condition: inf.Condition, nonce: str,
               document_text: str) -> Dict[str, str]:
    """Instruction, query and teacher-forced target for one cell.

    The target is condition-independent on purpose (§8.1): the compressed and
    uncompressed arms must read the *same* tokens, or an attention difference is
    partly a difference in what each model already generated.
    """
    if condition.task == "reconstruct":
        return {"query": "", "instruction": inf.RECONSTRUCTION_INSTRUCTION,
                "target": document_text}
    if condition.task == "conflict":
        return {"query": "", "instruction": inf.CONFLICT_TEMPLATE.format(nonce=nonce),
                "target": nonce}
    if condition.task == "qa":
        gold = sample["answers"][0] if sample.get("answers") else ""
        return {"query": sample["query"], "instruction": "", "target": gold}
    # qa_conflict: the real question is present and the instruction contradicts it,
    # which is the Level B form of SeleCom's conflict prompt.
    return {"query": sample["query"],
            "instruction": " " + inf.CONFLICT_TEMPLATE.format(nonce=nonce),
            "target": nonce}


# --------------------------------------------------------------------------------------
# Statistics reduction
# --------------------------------------------------------------------------------------
def reduce_statistics(collected: inf.CollectedAttention, max_steps: int) -> Dict[str, np.ndarray]:
    """Collapse ``[L, H, S, G]`` to what the figures and tests need, and no more.

    Head resolution is kept (§8.5 asks whether a handful of heads drives the
    effect) and the target-step axis is kept separately (§8.4 figure 3), but the
    product of the two is not stored: at 200 HotpotQA rows times ten conditions
    that would be hundreds of gigabytes for a plot nobody makes.
    """
    import warnings

    mass, density = collected.mass, collected.density

    def pad_steps(x: np.ndarray) -> np.ndarray:
        out = np.full((max_steps,) + x.shape[1:], np.nan, dtype=np.float32)
        n = min(max_steps, x.shape[0])
        out[:n] = x[:n]
        return out

    # A group absent from this condition (no document in the no-memory control, no
    # query in Level A) reduces over an all-NaN slice; NaN is the right answer and
    # the warning would fire once per layer per row.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        top = np.zeros(collected.mass.shape[:2] + (len(inf.GROUPS),), dtype=np.float32)
        for g in range(len(inf.GROUPS)):
            top[..., g] = (collected.top_group == g).mean(axis=2)
        return {
            "mass_lhg": np.nanmean(mass, axis=2).astype(np.float16),
            "density_lhg": np.nanmean(density, axis=2).astype(np.float16),
            "mass_sg": pad_steps(np.nanmean(mass, axis=(0, 1))).astype(np.float16),
            "density_sg": pad_steps(np.nanmean(density, axis=(0, 1))).astype(np.float16),
            "qk_mean_lhg": np.nanmean(collected.qk_mean, axis=2).astype(np.float16),
            "qk_max_lhg": np.nanmax(collected.qk_max, axis=2).astype(np.float16),
            "qk_mean_sg": pad_steps(np.nanmean(collected.qk_mean, axis=(0, 1))).astype(np.float16),
            "k_norm_lhg": collected.k_norm.astype(np.float16),
            "v_norm_lhg": collected.v_norm.astype(np.float16),
            "v_contrib_lg": np.nanmean(collected.v_contrib, axis=(1, 2)).astype(np.float16),
            "entropy_lh": np.nanmean(collected.entropy, axis=2).astype(np.float16),
            "hidden_norm_lg": collected.hidden_norm.astype(np.float32),
            "top_group_lg": top.mean(axis=1).astype(np.float32),
            "group_size_sg": pad_steps(collected.group_size).astype(np.float32),
        }


# --------------------------------------------------------------------------------------
# Backend agreement (§8.2)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def backend_agreement(lm, inputs_embeds: torch.Tensor) -> Dict[str, float]:
    """Eager must agree with the production backend, or the diagnostics describe
    a different model than the one every other number in this repository came from."""
    mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    original = lm.config._attn_implementation
    out = {}
    logits = {}
    for impl in ("eager", "sdpa"):
        try:
            lm.config._attn_implementation = impl
            logits[impl] = lm(inputs_embeds=inputs_embeds, attention_mask=mask,
                              use_cache=False).logits.float()
        except Exception as e:                       # pragma: no cover - backend probe
            out[f"{impl}_error"] = str(e)[:200]
        finally:
            lm.config._attn_implementation = original
    if "eager" in logits and "sdpa" in logits:
        diff = (logits["eager"] - logits["sdpa"]).abs()
        out["max_abs_logit_diff"] = float(diff.max())
        out["mean_abs_logit_diff"] = float(diff.mean())
        out["argmax_agreement"] = float(
            (logits["eager"].argmax(-1) == logits["sdpa"].argmax(-1)).float().mean())
    return out


@torch.no_grad()
def patch_fidelity(lm, inputs_embeds: torch.Tensor, group_index, target_positions
                   ) -> Dict[str, float]:
    """The recording copy of ``eager_attention_forward`` must be the original.

    It is a reimplementation, so "it looks the same" is not enough: the same input
    is run through both and the final logits compared.
    """
    mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
    plain = lm(inputs_embeds=inputs_embeds, attention_mask=mask, use_cache=False).logits.float()
    _, patched_out = inf.diagnostic_forward(lm, inputs_embeds, group_index, target_positions)
    patched = patched_out.logits.float()
    diff = (plain - patched).abs()
    return {"max_abs_logit_diff": float(diff.max()),
            "argmax_agreement": float((plain.argmax(-1) == patched.argmax(-1)).float().mean())}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def resummarise(run_dir: str) -> str:
    """Rebuild ``behavioral_metrics.json`` from ``examples.jsonl``.

    The per-example records hold everything the summary is derived from, so a new
    statistic can be added without another 100 GPU-minutes -- and without the
    temptation to leave it out because recomputing would be expensive.
    """
    path = os.path.join(run_dir, "examples.jsonl")
    examples = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    summary = summarise(examples)
    summary["resummarised_from"] = os.path.abspath(path)
    out = os.path.join(run_dir, "behavioral_metrics.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize_only", default=None,
                    help="recompute behavioral_metrics.json for an existing run directory")
    ap.add_argument("--level", choices=["A", "B"], default="A")
    ap.add_argument("--preset", default="pisco_hotpot")
    ap.add_argument("--rows", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--max_docs", type=int, default=None,
                    help="Level B only; defaults to the preset's K")
    ap.add_argument("--nonce_length", type=int, default=12)
    ap.add_argument("--max_new_tokens", type=int, default=None,
                    help="default: 192 for level A (reconstruction), 48 for level B")
    ap.add_argument("--max_target_steps", type=int, default=48,
                    help="teacher-forced target positions kept for the step axis")
    ap.add_argument("--figure_samples", type=int, default=3,
                    help="samples whose full attention tensor is saved for the figures")
    # /home is a shared disk that runs full, so the npz stacks live under
    # QURO_ROOT with the rest of the large artefacts; only the summary JSON and
    # the report are ever copied back into the repository.
    ap.add_argument("--out_root", default=os.path.join(
        paths.QURO_ROOT, "results", "full_compression_infeasibility"))
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--queries", default=None)
    ap.add_argument("--skip_generation", action="store_true",
                    help="mechanism statistics only; behavioural results are then absent")
    ap.add_argument("--prompt_style", choices=list(inf.PROMPT_STYLES),
                    default="pisco_question",
                    help="pisco_question keeps PISCO's trained scaffolding (and is "
                         "token-identical to D0/RG); selecom_literal drops the "
                         "'Question:' marker so the instruction reads as SeleCom wrote it")
    ap.add_argument("--system_prompt", choices=["pisco", "none"], default="pisco",
                    help="PISCO's system prompt orders the model to extract from the "
                         "documents, so 'none' is the control for whether a conflict "
                         "failure is that standing order rather than compression")
    args = ap.parse_args()

    if args.summarize_only:
        print(f"[done] {resummarise(args.summarize_only)}")
        return

    started = time.time()
    run_id = args.run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-level{args.level}"
    out_dir = os.path.abspath(os.path.join(args.out_root, run_id))
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} already exists; runs are never overwritten")
    os.makedirs(os.path.join(out_dir, "full_attention_examples"))
    print(f"[run] {out_dir}")

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    # -- model, cache, corpus: the repository's own loaders ---------------------
    cfg = get_config(args.preset)
    apply_arm(cfg, "P")
    cfg.generator.lora_init = "frozen"
    cfg.generator.attn_implementation = "eager"
    if args.max_docs is not None:
        cfg.data.max_docs = args.max_docs
    cfg.revalidate()

    cache = LatentCache(cfg.data.cache_dir)
    cfg.readout.cache_hidden = cache.metadata.hidden_size
    stack, model = build_model(cfg, cache_hidden=cache.metadata.hidden_size)
    device = torch.device(args.device)
    lm = stack.lm.to(device).eval()
    tokenizer = stack.tokenizer
    if lm.config._attn_implementation != "eager":
        raise SystemExit(f"decoder loaded with {lm.config._attn_implementation!r}; "
                         "the diagnostics need eager attention")

    corpus_path = args.corpus or os.path.join(os.path.dirname(cfg.data.train_file),
                                              "corpus.jsonl")
    print(f"[data] corpus {corpus_path}")
    corpus = load_corpus(corpus_path)

    doc_max_tokens = int(cache.metadata.doc_max_length or 128)
    system_prompt = (model.prompt_builders["D0"].system_prompt
                     if args.system_prompt == "pisco" else None)
    factory = PromptFactory(tokenizer, lm, cache, corpus, stack.n_mem_tokens,
                            doc_max_tokens, style=args.prompt_style,
                            system_prompt=system_prompt)

    # -- samples and conditions -------------------------------------------------
    if args.level == "A":
        samples = level_a_samples(tokenizer, corpus, cache, args.rows, rng)
        conditions = inf.level_a_conditions()
        max_new_tokens = args.max_new_tokens or 192
    else:
        queries = args.queries or cfg.data.eval_files["dev"]
        samples = level_b_samples(queries, cache, args.rows,
                                  cfg.data.max_docs or 10, rng)
        conditions = inf.level_b_conditions()
        max_new_tokens = args.max_new_tokens or 48
    print(f"[data] {len(samples)} samples x {len(conditions)} conditions")

    for sample in samples:
        sample.update(inf.make_nonce(rng, tokenizer, args.nonce_length))

    # -- Task 0 audit: our rendering is the repository's PISCO prompt ------------
    audit = prompt_equivalence_audit(model, factory, samples[0], tokenizer, cache)
    print("[audit] " + json.dumps(audit))

    # -- backend checks ---------------------------------------------------------
    probe = factory.build(samples[0], conditions[0], "probe instruction", "", device)
    probe_packed = factory.assemble(probe, [tokenizer.eos_token_id], device)
    probe_groups = probe["groups"].with_targets([tokenizer.eos_token_id])
    checks = {
        "backend": backend_agreement(lm, probe_packed["inputs_embeds"]),
        "patch": patch_fidelity(lm, probe_packed["inputs_embeds"],
                                probe_groups.index_tensor(probe_packed["inputs_embeds"].size(1)),
                                torch.tensor([probe_groups.spans["output_history"][0]])),
    }
    print("[checks] " + json.dumps(checks))

    # -- the sweep --------------------------------------------------------------
    examples: List[Dict] = []
    arrays: Dict[str, List[np.ndarray]] = {}
    figure_ids = {s["sample_id"] for s in samples[: args.figure_samples]}

    for n, sample in enumerate(samples):
        for condition in conditions:
            record = run_cell(lm, tokenizer, factory, sample, condition, device,
                              max_new_tokens=max_new_tokens,
                              max_target_steps=args.max_target_steps,
                              skip_generation=args.skip_generation,
                              keep_full=sample["sample_id"] in figure_ids,
                              out_dir=out_dir)
            reduced = record.pop("_arrays")
            for key, value in reduced.items():
                arrays.setdefault(key, []).append(value)
            examples.append(record)
        if (n + 1) % 5 == 0 or n + 1 == len(samples):
            elapsed = time.time() - started
            print(f"[{n + 1}/{len(samples)}] {elapsed / (n + 1):.1f}s/sample "
                  f"({elapsed / 60:.1f} min elapsed)")

    # -- write ------------------------------------------------------------------
    with open(os.path.join(out_dir, "examples.jsonl"), "w", encoding="utf-8") as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    keys = [str(e["sample_id"]) for e in examples]
    condition_names = [str(e["condition"]) for e in examples]
    attention_keys = ("mass_lhg", "density_lhg", "mass_sg", "density_sg",
                      "entropy_lh", "top_group_lg", "group_size_sg")
    np.savez_compressed(
        os.path.join(out_dir, "grouped_attention_stats.npz"),
        sample_id=np.array(keys), condition=np.array(condition_names),
        groups=np.array(inf.GROUPS),
        **{k: np.stack(arrays[k]) for k in attention_keys if k in arrays})
    norm_keys = ("qk_mean_lhg", "qk_max_lhg", "qk_mean_sg", "k_norm_lhg",
                 "v_norm_lhg", "v_contrib_lg", "hidden_norm_lg")
    np.savez_compressed(
        os.path.join(out_dir, "norm_logit_stats.npz"),
        sample_id=np.array(keys), condition=np.array(condition_names),
        groups=np.array(inf.GROUPS),
        **{k: np.stack(arrays[k]) for k in norm_keys if k in arrays})

    behavioural = summarise(examples)
    with open(os.path.join(out_dir, "behavioral_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(behavioural, f, indent=2, ensure_ascii=False)

    import yaml

    with open(os.path.join(out_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"args": vars(args), "config": _jsonable(cfg)}, f,
                       allow_unicode=True, sort_keys=True)

    manifest = {
        "run_id": run_id, "level": args.level, "git": git_state(),
        "environment": environment(),
        "command": " ".join([sys.executable] + sys.argv),
        "wall_clock_seconds": round(time.time() - started, 1),
        "model": {"checkpoint": cfg.generator.name_or_path,
                  "base_model": paths.MISTRAL_PATH,
                  "dtype": cfg.generator.dtype,
                  "attn_implementation": lm.config._attn_implementation,
                  "adapter": getattr(lm, "active_adapters", lambda: None)()
                  if callable(getattr(lm, "active_adapters", None))
                  else getattr(lm, "active_adapter", None),
                  "n_mem_tokens": stack.n_mem_tokens},
        "cache": {"dir": cfg.data.cache_dir, "compressor": cache.metadata.compressor,
                  "latent_size": cache.metadata.latent_size,
                  "hidden_size": cache.metadata.hidden_size,
                  "compr_rate": cache.metadata.compr_rate,
                  "doc_max_length": cache.metadata.doc_max_length,
                  "storage": cache.storage, "num_documents": len(cache)},
        "data": {"corpus": corpus_path,
                 "queries": (args.queries or cfg.data.eval_files.get("dev")
                             if args.level == "B" else None),
                 "seed": args.seed, "rows": len(samples),
                 "max_docs": cfg.data.max_docs},
        "prompt": {"style": args.prompt_style, "system_prompt": system_prompt,
                   "groups": list(inf.GROUPS),
                   "reconstruction_instruction": inf.RECONSTRUCTION_INSTRUCTION,
                   "conflict_template": inf.CONFLICT_TEMPLATE,
                   "nonce_length": args.nonce_length},
        "generation": {"do_sample": False, "max_new_tokens": max_new_tokens,
                       "eos_token_id": tokenizer.eos_token_id,
                       "skipped": args.skip_generation},
        "conditions": [{"name": c.name, "document_kind": c.document_kind,
                        "task": c.task} for c in conditions],
        "latents_per_sample": {s["sample_id"]: len(s["doc_ids"]) * cache.metadata.latent_size
                               for s in samples},
        "prompt_equivalence_audit": audit,
        "numerical_checks": checks,
        "figure_samples": sorted(figure_ids),
        "selecom_reference": {
            "source": "SeleCom (WWW'26) §3.2.1 and Figure 2",
            "status": "conceptual reproduction",
            "note": "the paper does not state which layers, heads or target rows "
                    "Figure 2 aggregates, so the pre-registered definitions in "
                    "docs/SELECOM_FULL_COMPRESSION_INFEASIBILITY_INSTRUCTION.md §8 "
                    "are used instead of reverse-engineering a flattering one"},
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + json.dumps(behavioural.get("headline", {}), indent=2, ensure_ascii=False))
    print(f"\n[done] {out_dir}  ({(time.time() - started) / 60:.1f} min)")


def _jsonable(cfg) -> Dict:
    from dataclasses import asdict

    return json.loads(json.dumps(asdict(cfg), default=str))


def prompt_equivalence_audit(model, factory: PromptFactory, sample: Dict,
                             tokenizer, cache: LatentCache) -> Dict[str, object]:
    """Task 0: prove this script renders the repository's PISCO prompt, not a lookalike.

    With an empty instruction the rendering must be token-identical to
    ``PiscoPromptBuilder`` in ``D0`` (memory slots) and ``RG`` (raw text).  If it
    is not, every number downstream describes some other system and the run
    should not be trusted.
    """
    query = sample.get("query") or "What is the capital of France?"
    budget = len(sample["doc_ids"]) * cache.metadata.latent_size

    compressed = inf.token_groups(
        tokenizer, inf.render(tokenizer, inf.slot_string(tokenizer, budget,
                                                         factory.n_mem_tokens),
                              query=query, instruction=""))
    reference_d0 = model.prompt_builders["D0"].build(query, budget)

    documents = [factory.corpus[d] for d in sample["doc_ids"]]
    raw = inf.token_groups(
        tokenizer, inf.render(tokenizer, factory.document_text(sample["doc_ids"]),
                              query=query, instruction=""))
    reference_rg = model.prompt_builders["RG"].build(query, budget, documents)

    return {
        "note": "rendered with the canonical pisco_question style and PISCO's system "
                "prompt; the run's own --prompt_style/--system_prompt may differ, and "
                "this audit is about the code path, not that run's scaffolding",
        "D0_token_identical": compressed.input_ids == reference_d0.input_ids,
        "D0_slots_identical": (inf.memory_positions(tokenizer, compressed)
                               == reference_d0.slot_positions),
        "RG_token_identical": raw.input_ids == reference_rg.input_ids,
        "D0_n_tokens": len(compressed.input_ids),
        "RG_n_tokens": len(raw.input_ids),
        "group_sizes_D0": compressed.sizes(),
    }


def run_cell(lm, tokenizer, factory: PromptFactory, sample: Dict,
             condition: inf.Condition, device, max_new_tokens: int,
             max_target_steps: int, skip_generation: bool, keep_full: bool,
             out_dir: str) -> Dict:
    """One (sample, condition): free generation, then the teacher-forced diagnostic."""
    # The teacher-forced target is built from the *correct* document in every
    # condition (§8.1): compressed, raw and mismatched arms must read identical
    # tokens or an attention difference is partly a difference in what is being
    # predicted.  Leakage, by contrast, is scored against what was shown.
    correct_document = factory.document_text(sample["doc_ids"])
    texts = task_texts(sample, condition, sample["nonce"], correct_document)

    if condition.task in ("qa", "qa_conflict") and not texts["target"]:
        texts["target"] = tokenizer.eos_token or " "

    built = factory.build(sample, condition, texts["instruction"], texts["query"], device)
    groups = built["groups"]
    shown_document = built["shown_document_text"]

    target_ids = tokenizer(" " + texts["target"].strip(), add_special_tokens=False
                           )["input_ids"][:max_target_steps]
    if not target_ids:
        target_ids = [tokenizer.eos_token_id]

    record: Dict[str, object] = {
        "sample_id": sample["sample_id"], "row_id": sample.get("row_id"),
        "condition": condition.name, "document_kind": condition.document_kind,
        "task": condition.task, "doc_ids": sample["doc_ids"],
        "n_docs": len(sample["doc_ids"]), "nonce": sample["nonce"],
        "query": texts["query"], "instruction": texts["instruction"],
        "prompt_text": groups.text, "n_prompt_tokens": groups.n_prompt_tokens,
        "group_spans": {k: list(v) for k, v in groups.spans.items()},
        "group_sizes": groups.sizes(), "n_target_tokens": len(target_ids),
        "target_text": texts["target"][:2000],
    }

    # -- behaviour: free generation --------------------------------------------
    if not skip_generation:
        packed = factory.assemble(built, None, device, pad_side="left")
        with torch.no_grad():
            generated = lm.generate(
                inputs_embeds=packed["inputs_embeds"],
                attention_mask=packed["attention_mask"],
                max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id)
        prediction = tokenizer.decode(generated[0].tolist(), skip_special_tokens=True).strip()
        record["prediction"] = prediction
        if condition.task in ("conflict", "qa_conflict"):
            record["metrics"] = inf.score_conflict(
                prediction, sample["nonce"], shown_document,
                sample["nonce_token_ids"], tokenizer)
        elif condition.task == "reconstruct":
            # Against the correct document, so this doubles as the qualification
            # gate: correct memory must beat mismatched and absent memory.
            record["metrics"] = inf.score_reconstruction(prediction, correct_document)
            record["metrics"]["shown_copied_span"] = float(
                inf.longest_common_span(prediction, shown_document))
        else:
            from src.metrics import score

            record["metrics"] = score(prediction, sample.get("answers") or [""])
            record["metrics"]["longest_copied_span"] = float(
                inf.longest_common_span(prediction, shown_document))

    # -- mechanism: teacher-forced forward -------------------------------------
    packed = factory.assemble(built, target_ids, device, pad_side="right")
    total = packed["inputs_embeds"].size(1)
    with_targets = groups.with_targets(target_ids)
    target_positions = torch.arange(*with_targets.spans["output_history"])
    group_index = with_targets.index_tensor(total)

    collected, output = inf.diagnostic_forward(
        lm, packed["inputs_embeds"], group_index, target_positions, keep_full=keep_full)

    record["dominance"] = inf.dominance_ratios(collected)
    record["dominance_vs_query"] = inf.dominance_ratios(collected, denominator="query")
    record["heads"] = inf.head_dominance(collected)
    record["nll"] = _target_nll_from(output, target_positions, target_ids)

    if keep_full and collected.full_attention is not None:
        name = f"{sample['sample_id']}__{condition.name.replace('/', '-')}.npz"
        np.savez_compressed(
            os.path.join(out_dir, "full_attention_examples", name),
            attention=collected.full_attention,
            group_index=group_index.numpy(),
            target_positions=target_positions.numpy(),
            groups=np.array(inf.GROUPS),
            input_ids=np.array(with_targets.input_ids))
        record["full_attention_file"] = name

    record["_arrays"] = reduce_statistics(collected, max_target_steps)
    return record


def _target_nll_from(output, target_positions: torch.Tensor,
                     target_ids: Sequence[int]) -> Dict[str, float]:
    """Reuse the diagnostic forward's logits rather than paying for a second pass."""
    logits = output.logits[0].float()
    predict_from = (target_positions - 1).to(logits.device)
    selected = logits[predict_from]
    log_probs = torch.log_softmax(selected, dim=-1)
    ids = torch.tensor(list(target_ids), device=logits.device)
    chosen = log_probs[torch.arange(ids.numel(), device=logits.device), ids]
    return {"nll": float(-chosen.mean()), "first_token_logprob": float(chosen[0]),
            "n_target_tokens": int(ids.numel())}


def mechanism_behaviour(by_condition: Dict[str, List[Dict]]) -> Dict[str, object]:
    """Per-example correlations between the internal statistics and the failure (§10).

    A condition-level mean can show dominance and failure moving together across
    conditions while they are unrelated *within* a condition -- which would mean
    the dominance is a property of the prompt layout, not a cause of the failure.
    Only the row-level correlation distinguishes the two, so it is reported even
    where it comes out flat.
    """
    out: Dict[str, object] = {}
    pairs = [("dominance.density_ratio", "metrics.leading"),
             ("dominance.mass_ratio", "metrics.leading"),
             ("dominance.qk_gap", "metrics.leading"),
             ("dominance.density_ratio", "metrics.longest_copied_span"),
             ("dominance.qk_gap", "metrics.longest_copied_span"),
             ("heads.dominant_head_fraction", "metrics.leading"),
             ("group_sizes.document", "metrics.leading"),
             ("nll.nll", "metrics.leading")]

    def read(row: Dict, path: str) -> float:
        head, tail = path.split(".", 1)
        block = row.get(head) or {}
        value = block.get(tail)
        return float(value) if isinstance(value, (int, float)) else float("nan")

    for condition, rows in sorted(by_condition.items()):
        if not condition.endswith("conflict") or len(rows) < 8:
            continue
        block = {}
        for mechanism, behaviour in pairs:
            block[f"{mechanism} ~ {behaviour}"] = inf.spearman(
                [read(r, mechanism) for r in rows], [read(r, behaviour) for r in rows])
        out[condition] = block
    return out


def summarise(examples: Sequence[Dict]) -> Dict[str, object]:
    """Per-condition means, the qualification gate, and the paired contrasts."""
    by_condition: Dict[str, List[Dict]] = {}
    for row in examples:
        by_condition.setdefault(str(row["condition"]), []).append(row)

    summary: Dict[str, object] = {"n_examples": len(examples), "per_condition": {}}
    for name, rows in sorted(by_condition.items()):
        block: Dict[str, object] = {"n": len(rows)}
        metric_keys = sorted({k for r in rows for k in (r.get("metrics") or {})})
        for key in metric_keys:
            values = [float(r["metrics"][key]) for r in rows if r.get("metrics")]
            block[key] = round(float(np.mean(values)), 4) if values else None
        for group_key in ("dominance", "dominance_vs_query", "heads", "nll"):
            for key in sorted({k for r in rows for k in (r.get(group_key) or {})}):
                values = [float(r[group_key][key]) for r in rows
                          if r.get(group_key) and np.isfinite(r[group_key][key])]
                block[f"{group_key}.{key}"] = round(float(np.mean(values)), 4) if values else None
        block["mean_prompt_tokens"] = round(float(np.mean([r["n_prompt_tokens"] for r in rows])), 1)
        block["mean_document_tokens"] = round(
            float(np.mean([r["group_sizes"].get("document", 0) for r in rows])), 1)
        block["mean_instruction_tokens"] = round(
            float(np.mean([r["group_sizes"].get("instruction", 0) for r in rows])), 1)
        summary["per_condition"][name] = block

    def paired(a: str, b: str, path: str) -> Optional[Dict[str, float]]:
        rows_a = {r["sample_id"]: r for r in by_condition.get(a, [])}
        rows_b = {r["sample_id"]: r for r in by_condition.get(b, [])}
        shared = sorted(set(rows_a) & set(rows_b))
        if not shared:
            return None
        head, tail = path.split(".", 1)
        x = [rows_a[s].get(head, {}).get(tail, float("nan")) for s in shared]
        y = [rows_b[s].get(head, {}).get(tail, float("nan")) for s in shared]
        return inf.paired_bootstrap(x, y)

    summary["qualification_gate"] = {
        "reconstruct_memory_vs_mismatch_nll": paired(
            "memory/reconstruct", "mismatch/reconstruct", "nll.nll"),
        "reconstruct_memory_vs_none_nll": paired(
            "memory/reconstruct", "none/reconstruct", "nll.nll"),
        "qa_memory_vs_mismatch_nll": paired("memory/qa", "mismatch/qa", "nll.nll"),
        "qa_memory_vs_none_nll": paired("memory/qa", "none/qa", "nll.nll"),
        "reconstruct_memory_vs_mismatch_rouge": paired(
            "memory/reconstruct", "mismatch/reconstruct", "metrics.rouge_l"),
        "reconstruct_memory_vs_none_rouge": paired(
            "memory/reconstruct", "none/reconstruct", "metrics.rouge_l"),
        "qa_memory_vs_mismatch_substring": paired(
            "memory/qa", "mismatch/qa", "metrics.substring"),
        "qa_memory_vs_none_substring": paired("memory/qa", "none/qa", "metrics.substring"),
    }
    summary["conflict_contrast"] = {
        # The headline pairing: same decoder, same instruction, same nonce, the
        # document representation the only difference.
        "memory_vs_raw_leading": paired("memory/conflict", "raw/conflict", "metrics.leading"),
        "memory_vs_zero_leading": paired("memory/conflict", "zero/conflict", "metrics.leading"),
        "memory_vs_none_leading": paired("memory/conflict", "none/conflict", "metrics.leading"),
        "mismatch_vs_memory_leading": paired(
            "mismatch/conflict", "memory/conflict", "metrics.leading"),
        "memory_vs_raw_copied_span": paired(
            "memory/conflict", "raw/conflict", "metrics.longest_copied_span"),
        "memory_vs_raw_nonce_nll": paired("memory/conflict", "raw/conflict", "nll.nll"),
        "memory_vs_zero_nonce_nll": paired("memory/conflict", "zero/conflict", "nll.nll"),
        "memory_vs_none_nonce_nll": paired("memory/conflict", "none/conflict", "nll.nll"),
        "memory_vs_raw_density_ratio": paired(
            "memory/conflict", "raw/conflict", "dominance.density_ratio"),
        "memory_vs_raw_mass_ratio": paired(
            "memory/conflict", "raw/conflict", "dominance.mass_ratio"),
        "memory_vs_raw_qk_gap": paired("memory/conflict", "raw/conflict", "dominance.qk_gap"),
        "qa_memory_vs_raw_density_ratio": paired(
            "memory/qa", "raw/qa", "dominance.density_ratio"),
    }
    summary["mechanism_behaviour"] = mechanism_behaviour(by_condition)

    headline = {}
    for name in ("memory/conflict", "raw/conflict", "none/conflict", "zero/conflict",
                 "memory/qa_conflict", "raw/qa_conflict"):
        block = summary["per_condition"].get(name)
        if block:
            headline[name] = {k: block.get(k) for k in
                              ("exact", "leading", "mentions", "nonce_share",
                               "longest_copied_span", "dominance.mass_ratio",
                               "dominance.density_ratio", "dominance.qk_gap",
                               "nll.nll")}
    summary["headline"] = headline
    return summary


if __name__ == "__main__":
    main()
