"""Train and evaluate QuRO from a precomputed latent cache.

The offline compressor is deliberately absent from this file.  Training reads
``(m, h)`` tensors off disk, so what gets optimised is exactly what gets served,
and the "reusable precomputed representation" claim is exercised rather than
assumed.

Two training details carry weight and are easy to get wrong:

*Budget dropout.*  ``slots[:B]`` silently assumes the output slots are nested.
Sampling ``B`` per step makes that true, so one checkpoint serves every budget
instead of needing a separate run per point of the main table.

*Residual warm-up.*  The readout starts as attention-pooled PISCO (zero-initialised
residual).  A decaying penalty on the residual keeps it near that working solution
for the first few hundred steps, which is the practical fix for the slow
cross-attention convergence noted in ``QURO_EXPERIMENTAL_DESIGN.md`` §8.1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as config_module
from config import apply_arm, arm_label, get_config, parse_eval_files
from src import metrics
from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset, load_corpus, move_to_device
from src.distill import TeacherCache
from src.model import build_model


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cycle(loader):
    while True:
        yield from loader


def lr_lambda_factory(total, warmup_ratio):
    warmup = max(1, int(total * warmup_ratio))

    def schedule(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return schedule


def build_args():
    ap = argparse.ArgumentParser()
    # Derived from config.PRESETS rather than repeated: a hardcoded copy silently
    # hides every preset added after it was written.
    ap.add_argument("--preset", default="pisco_smoke", choices=sorted(config_module.PRESETS))
    ap.add_argument("--config_json", default=None,
                    help="rebuild a saved run config before applying explicit overrides")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--decoder_lr", type=float, default=None,
                    help="separate LR for the decoder LoRA; defaults to --lr")
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    # Distillation from the full-cache teacher.  The gap to P is what defines the
    # problem, so P is the teacher; the gold CE stays because P is wrong often
    # enough that replacing the labels would inherit its mistakes too.
    ap.add_argument("--teacher_logits", default=None,
                    help="precomputed teacher distributions from "
                         "scripts/build_teacher_logits.py")
    ap.add_argument("--kd_weight", type=float, default=0.0,
                    help="weight on the KL term; 0 disables distillation")
    ap.add_argument("--warm_start", action="store_true",
                    help="with --resume_from: load weights only, and build a fresh "
                         "optimiser and schedule instead of continuing the old run")
    ap.add_argument("--eval_every", type=int, default=None,
                    help="validate every N steps on a small dev slice; 0 disables")
    ap.add_argument("--eval_every_samples", type=int, default=None)
    ap.add_argument("--select_metric", choices=["em", "substring", "f1"], default=None,
                    help="which metric picks checkpoint_best.pt; fix it before the run")
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume_from", default=None)
    ap.add_argument("--baseline_run", default=None,
                    help="start a NEW full-length P/R experiment from this P run's "
                         "config.json and checkpoint_last.pt; requires --out_dir")

    ap.add_argument("--readout", choices=["quro", "pisco_direct", "similarity_topb", "pisco_residual"], default=None)
    ap.add_argument("--output_query_mode",
                    choices=["agnostic", "agnostic_matched", "add", "film", "concat", "xattn"],
                    default=None)
    # Sets output_query_mode and cosine_prior together, so an arm cannot be
    # half-specified the way the historical A runs were (HANDOFF.md §3 W1).
    ap.add_argument("--arm", choices=sorted(config_module.ARMS), default=None)
    ap.add_argument("--agnostic_param_matched", action="store_true",
                    help="A arms use agnostic_matched, so A and C have equal parameters")
    ap.add_argument("--budget", type=int, default=None, help="B_max and the fixed eval budget")
    ap.add_argument("--budget_buckets", default=None, help="comma-separated discrete B values")
    ap.add_argument("--no_budget_dropout", action="store_true")
    ap.add_argument("--readout_blocks", type=int, default=None)
    ap.add_argument("--d_readout", type=int, default=None)
    # Legacy: equivalent to --readout_output_mode delta_only (and, historically,
    # it also swapped out_proj to the default init -- pass --out_proj_init default
    # to reproduce that exactly).  See docs/HANDOFF.md §3 W2.
    ap.add_argument("--no_residual_readout", action="store_true")
    ap.add_argument("--readout_output_mode",
                    choices=["full", "pool_only", "delta_only"], default=None)
    ap.add_argument("--out_proj_init", choices=["zeros", "default"], default=None)
    # "frozen query encoder" was never true for kind=generator: the encoder and
    # the decoder are one object, so the decoder's LoRA updates move the query
    # representation.  fixed_adapter makes it an actual fixed function; the
    # default keeps the historical behaviour so old runs stay reproducible.
    ap.add_argument("--query_representation",
                    choices=["shared_current", "fixed_adapter"], default=None)
    ap.add_argument("--no_cosine_prior", action="store_true")
    ap.add_argument("--prior_mode", choices=["rank", "shared"], default=None)
    ap.add_argument("--tau_init", type=float, default=None)
    ap.add_argument("--without_document_source", action="store_true")
    ap.add_argument("--adaptive_budget", action="store_true")

    # D4/D5 compress the question into embeddings instead of sending its text.
    # D4 keeps PISCO's scaffolding; D5 keeps nothing but the two slot groups, so
    # everything the decoder reads is a compressed vector.
    ap.add_argument("--decoder_input_mode",
                    choices=["D0", "D1", "D2", "D3", "D4", "D5", "AG", "RG"],
                    default=None)
    ap.add_argument("--query_tokens", type=int, default=None,
                    help="embeddings the question is compressed into for D4/D5; "
                         "HotpotQA questions average ~23 tokens, so 6 is about 4x")
    ap.add_argument("--query_text_dropout", type=float, default=None)
    ap.add_argument("--generator_lora_init", choices=["pisco", "random", "frozen"], default=None)
    ap.add_argument("--generator_path", default=None,
                    help="checkpoint for the generator. Must match the compressor that "
                         "produced the cache: COCOM latents come from COCOM's adapted "
                         "Mistral and PISCO's decoder was never trained to read them.")
    ap.add_argument("--generator_n_mem", type=int, default=None,
                    help="slots per document block; required for COCOM v1, which "
                         "exposes neither n_mem_tokens nor doc_max_length")

    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--corpus", nargs="*", default=None,
                    help="corpus.jsonl files; required only by the uncompressed RG baseline")
    ap.add_argument("--disable_generator_adapter", action="store_true",
                    help="run plain Mistral without PISCO's compression adapter -- "
                         "the honest backbone for the AG/RG rows")
    ap.add_argument("--train_file", default=None)
    ap.add_argument("--eval_files", default=None)
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--eval_max_samples", type=int, default=None)
    ap.add_argument("--num_workers", type=int, default=None,
                    help="DataLoader prefetch workers; 0 reads the cache inline")
    ap.add_argument("--eval_budgets", default=None,
                    help="comma-separated budgets to evaluate, e.g. 4,8")
    ap.add_argument("--eval_input_modes", default=None,
                    help="comma-separated decoder input modes to evaluate, e.g. D0,D1")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--query_control", action="store_true",
                    help="also evaluate with a mismatched query")
    ap.add_argument("--doc_control", action="store_true",
                    help="also evaluate with mismatched documents: the gap to the "
                         "normal run is what the evidence path is actually worth")
    ap.add_argument("--dump_attn", action="store_true")
    return ap.parse_args()


def apply_overrides(cfg, args):
    simple = [
        ("steps", cfg.train), ("batch_size", cfg.train), ("lr", cfg.train),
        ("grad_accum", cfg.train), ("seed", cfg.train), ("device", cfg.train),
        ("decoder_lr", cfg.train), ("eval_every", cfg.train),
        ("query_tokens", cfg.decoder),
        ("eval_every_samples", cfg.train), ("select_metric", cfg.train),
        ("out_dir", cfg.train), ("resume_from", cfg.train),
        ("eval_max_samples", cfg.train), ("num_workers", cfg.train),
        ("d_readout", cfg.readout), ("cache_dir", cfg.data),
        ("train_file", cfg.data), ("max_docs", cfg.data),
    ]
    for name, target in simple:
        value = getattr(args, name)
        if value is not None:
            setattr(target, name, value)
    if args.readout:
        cfg.readout.kind = args.readout
    if args.output_query_mode:
        cfg.readout.output_query_mode = args.output_query_mode
    # Applied after --readout/--output_query_mode but before the individual knobs,
    # so a deliberate one-off override still works.  Whatever wins is re-derived by
    # arm_label() into the run record, so a mismatch surfaces in the results file.
    if args.arm:
        apply_arm(cfg, args.arm, param_matched=args.agnostic_param_matched)
    if args.budget is not None:
        cfg.readout.max_budget = args.budget
    if args.budget_buckets:
        cfg.readout.budget_buckets = [int(x) for x in args.budget_buckets.split(",")]
    if args.readout_blocks is not None:
        cfg.readout.num_blocks = args.readout_blocks
    if args.no_residual_readout:
        cfg.readout.residual_readout = False
        cfg.readout.output_mode = "delta_only"
    if args.readout_output_mode:
        cfg.readout.output_mode = args.readout_output_mode
        cfg.readout.residual_readout = (args.readout_output_mode == "full")
    if args.out_proj_init:
        cfg.readout.out_proj_init = args.out_proj_init
    if args.query_representation:
        cfg.query_encoder.representation = args.query_representation
    if args.no_cosine_prior:
        cfg.readout.cosine_prior = False
    if args.prior_mode:
        cfg.readout.prior_mode = args.prior_mode
    if args.tau_init is not None:
        cfg.readout.tau_init = args.tau_init
    if args.without_document_source:
        cfg.readout.add_document_source = False
    if args.adaptive_budget:
        cfg.readout.adaptive_budget = True
    if args.no_budget_dropout:
        cfg.train.budget_dropout = False
    if args.decoder_input_mode:
        cfg.decoder.input_mode = args.decoder_input_mode
    if args.query_text_dropout is not None:
        cfg.decoder.query_text_dropout = args.query_text_dropout
    if args.generator_lora_init:
        cfg.generator.lora_init = args.generator_lora_init
    if args.generator_path:
        cfg.generator.name_or_path = args.generator_path
    if args.generator_n_mem is not None:
        cfg.generator.n_mem_tokens = args.generator_n_mem
    if args.eval_files:
        cfg.data.eval_files = parse_eval_files(args.eval_files)
    cfg.data.prefer_teacher_output = cfg.train.prefer_teacher_output
    return cfg.revalidate()


def build_loaders(cfg, tokenizer, query_tokenizer, collator, query_control,
                  doc_control=False, corpus=None):
    train_set = QuRODataset(cfg.data.train_file, tokenizer, cfg.data,
                            query_tokenizer=query_tokenizer)
    # Loading is not free at scale: one batch pulls B*K*m*h*2 bytes out of the
    # memmap -- 21 MB at m=32 -- and with num_workers=0 that read blocks the
    # training step.  Measured at m=32 the GPUs idle waiting on it.
    train_loader = DataLoader(train_set, batch_size=cfg.train.batch_size, shuffle=True,
                              collate_fn=collator, drop_last=True,
                              num_workers=cfg.train.num_workers,
                              pin_memory=cfg.train.num_workers > 0,
                              persistent_workers=cfg.train.num_workers > 0,
                              prefetch_factor=4 if cfg.train.num_workers > 0 else None)
    evals = {}
    for name, path in cfg.data.resolved_eval_files().items():
        # (variant, readout-query shift, decoder-query shift, document shift).
        # "mismatch-q" now moves *only* the readout's question: the decoder still
        # gets the right one, so a drop is attributable to the readout rather than
        # to the decoder being asked something else.  "mismatch-q-both" is the old
        # joint shift, kept for comparability with the historical runs
        # (docs/HANDOFF.md §3 W4).
        variants = [(name, 0, 0, 0)]
        if query_control:
            variants.append((name + "/mismatch-q", 1, 0, 0))
            variants.append((name + "/mismatch-q-both", 1, 1, 0))
        if doc_control:
            variants.append((name + "/mismatch-doc", 0, 0, 1))
        for variant, rq_shift, dq_shift, d_shift in variants:
            dataset = QuRODataset(path, tokenizer, cfg.data, query_tokenizer=query_tokenizer,
                                  readout_query_shift=rq_shift, decoder_query_shift=dq_shift,
                                  document_shift=d_shift,
                                  limit=cfg.train.eval_max_samples, corpus=corpus)
            evals[variant] = DataLoader(dataset, batch_size=cfg.train.eval_batch_size,
                                        shuffle=False, collate_fn=collator)
    return train_set, train_loader, evals


@torch.no_grad()
def evaluate(model, loader, device, max_new_tokens, budget=None, dump_attn_path=None):
    model.eval()
    rows, attention, source_tokens, readout_tokens = [], [], 0, 0
    prompt_tokens = 0
    for batch in loader:
        batch = move_to_device(batch, device)
        predictions = model.generate_answer(batch, max_new_tokens=max_new_tokens, budget=budget)
        result = model.readout_cached(batch, budget=budget,
                                      return_attn=bool(dump_attn_path))
        # Count the slots the prompt really carries: AG has none and RG none
        # either, so xi_eff stays meaningful across every row of the table.
        # What the decoder actually prefills, counted through the same code path
        # that built the prompts.  This is the cost axis the method trades against:
        # a few points of accuracy are cheap if the prefill is an order of
        # magnitude shorter.
        for prompt in model.build_prompts(batch, result["soft_token_mask"], training=False):
            prompt_tokens += len(prompt.input_ids)
            readout_tokens += len(prompt.slot_positions)
        if "source_token_counts" in batch:
            counts = batch["source_token_counts"] * batch["document_mask"]
            source_tokens += int(counts.sum())
        if dump_attn_path and result["aux"].get("attention") is not None:
            attention.append(result["aux"]["attention"].float().cpu())
        decoder_queries = batch.get("queries", [])
        readout_queries = batch.get("readout_queries", [])
        swapped_answers = batch.get("readout_query_answers", [])
        for i, (item, prediction) in enumerate(zip(batch["raw"], predictions)):
            golds = item.get("answers") or [item["answer"]]
            row = {"id": item["id"], "query": item["query"], "golds": golds,
                   "pred": prediction, **metrics.score(prediction, golds)}
            # Under a mismatch control the two routes carry different questions, so
            # record both rather than only the row's original one.
            if i < len(decoder_queries) and decoder_queries[i] != item["query"]:
                row["decoder_query"] = decoder_queries[i]
            if i < len(readout_queries) and readout_queries[i] != item["query"]:
                row["readout_query"] = readout_queries[i]
                # A swapped question that shares this row's answer is not a control.
                shared = {a.strip().lower() for a in (swapped_answers[i] or []) if a}
                row["mismatch_shares_answer"] = bool(
                    shared & {g.strip().lower() for g in golds if g})
            rows.append(row)
    if attention:
        torch.save(attention, dump_attn_path)
    aggregate = metrics.aggregate(rows)
    # Always carry the "ignore the input and answer the same thing every time"
    # floor alongside the score, so a number can never be read without it.
    floor = metrics.constant_baseline([r["golds"] for r in rows])
    aggregate["constant_baseline_em"] = floor["em"]
    aggregate["constant_baseline_f1"] = floor["f1"]
    aggregate["constant_baseline_substring"] = floor["substring"]
    aggregate["constant_baseline_answer"] = floor["answer"]
    aggregate["em_above_constant"] = aggregate["em"] - floor["em"]
    aggregate["source_tokens"] = source_tokens
    aggregate["readout_tokens"] = readout_tokens
    aggregate["decoder_input_tokens"] = prompt_tokens
    aggregate["mean_decoder_input_tokens"] = prompt_tokens / max(1, len(rows))
    # Generator-side effective compression: the only ratio that makes two systems
    # comparable, because it is measured where the cost is actually paid.
    aggregate["xi_eff"] = (source_tokens / readout_tokens) if readout_tokens else None
    return aggregate, rows


def run_evaluations(model, loaders, device, cfg, args, cache):
    out_dir = cfg.train.out_dir
    budgets = ([int(x) for x in args.eval_budgets.split(",")] if args.eval_budgets
               else [cfg.readout.max_budget])
    modes = (args.eval_input_modes.split(",") if args.eval_input_modes
             else [cfg.decoder.input_mode])
    result = {
        "version": config_module.__version__,
        "tag": args.tag,
        "readout": cfg.readout.kind,
        "output_query_mode": cfg.readout.output_query_mode,
        "cosine_prior": cfg.readout.cosine_prior,
        "arm": arm_label(cfg),
        # Not cosmetic: under shared_current the query representation drifts with
        # the decoder LoRA, so two runs differing only in this flag are not the
        # same system and must not be pooled.
        "query_representation": cfg.query_encoder.representation,
        "query_adapter_hash": getattr(model.query_encoder, "query_adapter_hash", None),
        "kd_weight": args.kd_weight,
        "teacher_logits": args.teacher_logits,
        "readout_output_mode": cfg.readout.output_mode,
        "out_proj_init": cfg.readout.out_proj_init,
        "generator_lora_init": cfg.generator.lora_init,
        "train_decoder_input_mode": cfg.decoder.input_mode,
        "query_text_dropout": cfg.decoder.query_text_dropout,
        "offline_m": cache.metadata.latent_size,
        "offline_compressor": cache.metadata.compressor,
        "offline_compr_rate": cache.metadata.compr_rate,
        "baseline_initialization": getattr(model, "baseline_initialization", None),
        "budget_semantics": ("all_cached_latents" if cfg.readout.kind in
                             {"pisco_direct", "pisco_residual"} else "output_budget"),
        "metrics": {},
    }
    original_mode = model.decoder_input_mode
    for mode in modes:
        model.decoder_input_mode = mode
        for budget in budgets:
            for name, loader in loaders.items():
                key = f"{name}|{mode}|B={budget}"
                dump = (os.path.join(out_dir, f"attention_{name.replace('/', '_')}_{mode}_B{budget}.pt")
                        if args.dump_attn and "/" not in name else None)
                aggregate, rows = evaluate(model, loader, device,
                                           cfg.train.gen_max_new_tokens, budget, dump)
                result["metrics"][key] = aggregate
                filename = f"predictions_{name.replace('/', '_')}_{mode}_B{budget}.json"
                with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
                    # Every row, not a prefix: paired significance tests between
                    # arms need the whole split, and a 1.5-point difference is
                    # unresolvable at n=500 (McNemar p=0.44) but reachable at 2000.
                    json.dump(rows, f, ensure_ascii=False, indent=2)
                print(f"[eval] {key}: EM={aggregate['em']:.2%} F1={aggregate['f1']:.3f} "
                      f"sub={aggregate['substring']:.2%} "
                      f"(floor {aggregate['constant_baseline_em']:.2%} "
                      f"-> {aggregate['em_above_constant']:+.2%}) "
                      f"prefill={aggregate['mean_decoder_input_tokens']:.0f} tok "
                      f"xi_eff={aggregate['xi_eff']}")
    model.decoder_input_mode = original_mode
    with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def main():
    args = build_args()
    baseline_checkpoint = None
    if args.baseline_run:
        if args.config_json:
            raise SystemExit("choose --baseline_run or --config_json, not both")
        if args.resume_from or args.warm_start:
            raise SystemExit("--baseline_run starts a new run; do not combine with resume/warm_start")
        if not args.out_dir or os.path.realpath(args.out_dir) == os.path.realpath(args.baseline_run):
            raise SystemExit("--baseline_run requires a separate --out_dir")
        if os.path.exists(os.path.join(args.out_dir, "config.json")):
            raise SystemExit("baseline experiment output already exists; use a fresh --out_dir")
        with open(os.path.join(args.baseline_run, "config.json"), encoding="utf-8") as f:
            saved_cfg = json.load(f)
        cfg = config_module.Config()
        for name, values in saved_cfg.items():
            setattr(cfg, name, type(getattr(cfg, name))(**values))
        if cfg.readout.kind != "pisco_direct":
            raise SystemExit("--baseline_run must point to a P (pisco_direct) run")
        cfg.train.resume_from = None
        cfg.train.budget_dropout = False
        cfg.train.residual_weight = 0.0
        cfg.generator.lora_init = "frozen"
        cfg.query_encoder.representation = "fixed_adapter"
        cfg.readout.d_readout = 256
        cfg.readout.num_blocks = 1
        cfg.readout.dropout = 0.0
        cfg.readout.kind = "pisco_residual"
        cfg.readout.cosine_prior = False
        cfg = apply_overrides(cfg, args)
        if cfg.readout.kind not in {"pisco_direct", "pisco_residual"}:
            raise SystemExit("baseline experiment supports only P and R")
        if cfg.decoder.input_mode != "D0" or cfg.decoder.query_text_dropout != 0:
            raise SystemExit("baseline experiment requires D0 and query_text_dropout=0")
        if args.teacher_logits or args.kd_weight:
            raise SystemExit("first baseline experiment uses CE only; KD is disabled")
        if cfg.readout.adaptive_budget or cfg.train.budget_dropout:
            raise SystemExit("baseline experiment keeps all latent tokens")
        baseline_checkpoint = os.path.join(args.baseline_run, "checkpoint_last.pt")
    else:
        cfg = get_config(args.preset)
        if args.config_json:
            with open(args.config_json, encoding="utf-8") as f:
                for name, values in json.load(f).items():
                    setattr(cfg, name, type(getattr(cfg, name))(**values))
        cfg = apply_overrides(cfg, args)
    os.makedirs(cfg.train.out_dir, exist_ok=True)
    set_seed(cfg.train.seed)
    device = pick_device(cfg.train.device)

    corpus = load_corpus(args.corpus) if args.corpus else None
    if corpus:
        print(f"[data] loaded {len(corpus)} document texts for the RG baseline")

    cache = LatentCache(cfg.data.cache_dir)
    cfg.readout.cache_hidden = cache.metadata.hidden_size
    print(f"[cache] {cache.metadata.compressor}: {len(cache)} docs, "
          f"m={cache.metadata.latent_size}, h={cache.metadata.hidden_size}")

    stack, model = build_model(cfg, cache_hidden=cache.metadata.hidden_size)
    if args.disable_generator_adapter:
        stack.lm.disable_adapters()
        print("[generator] PISCO adapters disabled: plain Mistral-7B-Instruct-v0.2")
    model.to(device)
    stack.lm.to(device)
    if baseline_checkpoint:
        from src.refinement import initialize_from_pisco
        model.baseline_initialization = initialize_from_pisco(model, baseline_checkpoint)
        print(f"[baseline] {model.baseline_initialization}")
    cfg.to_json(os.path.join(cfg.train.out_dir, "config.json"))
    print(cfg.summary())
    report = model.parameter_report()
    counts = {k: round(v / 1e6, 2) for k, v in report.items() if isinstance(v, int)}
    provenance = {k: v for k, v in report.items() if not isinstance(v, int)}
    print(f"[params] {json.dumps(counts)} (M) | {json.dumps(provenance)}")

    collator = QuROCollator(
        cache, pad_id=model.pad_id,
        query_pad_id=getattr(stack.query_tokenizer, "pad_token_id", model.pad_id),
        max_docs=cfg.data.max_docs,
        require_budget_labels=cfg.readout.adaptive_budget and not args.eval_only)
    train_set, train_loader, eval_loaders = build_loaders(
        cfg, stack.tokenizer, stack.query_tokenizer, collator, args.query_control,
        args.doc_control, corpus)
    print(f"[data] train={len(train_set)} device={device}")

    if baseline_checkpoint and cfg.readout.kind == "pisco_residual":
        from src.refinement import verify_pisco_identity
        # No shuffled loader iteration: the check must not consume training RNG.
        state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        probe = collator([train_set[i] for i in range(min(2, len(train_set)))])
        check = verify_pisco_identity(model, move_to_device(probe, device))
        torch.random.set_rng_state(state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        with open(os.path.join(cfg.train.out_dir, "baseline_identity.json"), "w") as f:
            json.dump({**model.baseline_initialization, **check}, f, indent=2)
        print(f"[baseline identity] {check}")

    if args.eval_only:
        if cfg.train.resume_from:
            model.load(cfg.train.resume_from)
        run_evaluations(model, eval_loaders, device, cfg, args, cache)
        return

    params = model.trainable_parameters()
    if not params:
        raise RuntimeError("nothing is trainable; check readout kind and generator_lora_init")
    # Group by module, not by name: the decoder LoRA starts from PISCO's trained
    # weights while the readout starts from scratch, so one rate for both is a
    # choice rather than a default.  Identity comparison, because the same tensor
    # must not land in two groups -- AdamW would then step it twice.
    decoder_params = [p for p in model.lm.parameters() if p.requires_grad]
    decoder_ids = {id(p) for p in decoder_params}
    readout_params = [p for p in params if id(p) not in decoder_ids]
    decoder_lr = cfg.train.lr if cfg.train.decoder_lr is None else cfg.train.decoder_lr
    groups = [{"params": readout_params, "lr": cfg.train.lr, "name": "readout"}]
    if decoder_params:
        groups.append({"params": decoder_params, "lr": decoder_lr, "name": "decoder_lora"})
    if sum(len(g["params"]) for g in groups) != len(params):
        raise RuntimeError("parameter groups do not partition the trainable set")
    print(f"[optim] readout lr={cfg.train.lr} ({len(readout_params)} tensors) | "
          f"decoder lr={decoder_lr} ({len(decoder_params)} tensors)")
    optimizer = torch.optim.AdamW(groups, lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg.train.steps, cfg.train.warmup_ratio))
    start_step = 0
    best = {"metric": float("-inf"), "step": None}
    if cfg.train.resume_from:
        # Two different things were both called "resume".  Continuing a run has to
        # restore the optimiser, the schedule and the best-so-far, or the next
        # validation overwrites a better checkpoint with a worse one.  Starting a
        # new run from trained weights must NOT restore them, or the freshly
        # requested learning rates are silently overwritten by the saved ones.
        if args.warm_start:
            model.load(cfg.train.resume_from)
            print(f"[init] warm start from {cfg.train.resume_from}: weights only, "
                  f"fresh optimiser and schedule")
        else:
            try:
                _, _, saved = model.load(cfg.train.resume_from,
                                         optimizer=optimizer, scheduler=scheduler)
            except ValueError as error:
                # Old checkpoints were written with a single parameter group; the
                # grouped optimiser has two, and torch rejects the mismatch.
                raise SystemExit(
                    f"cannot resume optimiser state from {cfg.train.resume_from}: "
                    f"{error}\nThis checkpoint predates grouped learning rates. "
                    "Pass --warm_start to load the weights and start a fresh "
                    "optimiser, or resume without --decoder_lr on the old code.")
            start_step = int(saved or 0)
            best_path = os.path.join(os.path.dirname(cfg.train.resume_from),
                                     "best_checkpoint.json")
            if os.path.exists(best_path):
                with open(best_path, encoding="utf-8") as f:
                    best = json.load(f)
                print(f"[init] resumed at step {start_step}; best so far "
                      f"{best.get('metric')} at step {best.get('step')}")

    # Interval validation on a small, fixed dev slice.  It exists to answer
    # "was 3000 steps too few, or is the capacity not there" -- a question the
    # final-step-only protocol cannot answer, because a run that peaked at 1500
    # and then overfitted looks identical to one that never got there.  The slice
    # is small and fixed: it ranks checkpoints, it is not a reportable number, and
    # the full dev evaluation at the end is unchanged.
    validation = None
    if cfg.train.eval_every > 0:
        clean = [name for name in eval_loaders if "/" not in name]
        if not clean:
            raise RuntimeError("eval_every needs an eval split without a control suffix")
        name = clean[0]
        small = QuRODataset(cfg.data.resolved_eval_files()[name], stack.tokenizer, cfg.data,
                            query_tokenizer=stack.query_tokenizer,
                            limit=cfg.train.eval_every_samples, corpus=corpus)
        validation = (name, DataLoader(small, batch_size=cfg.train.eval_batch_size,
                                       shuffle=False, collate_fn=collator))
        print(f"[val] every {cfg.train.eval_every} steps on {len(small)} rows of "
              f"{name}, selecting on {cfg.train.select_metric}")

    teacher = None
    if args.teacher_logits:
        if args.kd_weight <= 0:
            raise SystemExit("--teacher_logits given but --kd_weight is 0")
        teacher = TeacherCache.load(args.teacher_logits)
        meta = teacher.meta
        if meta.get("split") != "train":
            raise SystemExit(
                f"teacher cache was built from {meta.get('split')!r}; distilling on "
                "anything but train routes evaluation data into the student")
        if meta.get("cache_dir") != cfg.data.cache_dir:
            raise SystemExit(
                f"teacher read latents from {meta.get('cache_dir')} but this run "
                f"reads {cfg.data.cache_dir}; they are different documents")
        if meta.get("source_file") != cfg.data.train_file:
            raise SystemExit(
                f"teacher was built on {meta.get('source_file')} but this run trains "
                f"on {cfg.data.train_file}; example ids would not correspond")
        print(f"[kd] teacher={meta['teacher']} step={meta['teacher_step']} "
              f"B={meta['teacher_budget']} T={meta['temperature']} "
              f"top_k={meta['top_k']} | {meta['examples']} examples, "
              f"{meta['positions']} positions | weight={args.kd_weight}")

    if args.baseline_run:
        # P and R construct different modules. Reset after construction so this
        # does not change the shuffled training examples in a matched comparison.
        set_seed(cfg.train.seed)
    iterator = cycle(train_loader)
    model.train()
    rng = random.Random(cfg.train.seed)
    buckets = list(cfg.readout.budget_buckets)
    log_path = os.path.join(cfg.train.out_dir, "train_log.jsonl")
    started = time.time()

    with open(log_path, "a", encoding="utf-8") as log:
        for step in range(start_step, cfg.train.steps):
            # Decays to zero so the trust region never constrains the final model.
            residual_weight = cfg.train.residual_weight * max(
                0.0, 1.0 - step / max(1, cfg.train.residual_warmup_steps))
            budget = (rng.choice(buckets) if cfg.train.budget_dropout
                      else cfg.readout.max_budget)

            optimizer.zero_grad(set_to_none=True)
            totals = {"loss": 0.0, "qa_loss": 0.0, "mean_budget": 0.0,
                      "residual_penalty": 0.0, "kd_loss": 0.0}
            for _ in range(max(1, cfg.train.grad_accum)):
                batch = move_to_device(next(iterator), device)
                output = model(batch, budget=budget, residual_weight=residual_weight,
                               teacher=teacher, kd_weight=args.kd_weight)
                if not bool(torch.isfinite(output["loss"])):
                    raise FloatingPointError(f"non-finite loss at step {step + 1}; no update applied")
                (output["loss"] / cfg.train.grad_accum).backward()
                for key in totals:
                    if key in output:
                        totals[key] += float(output[key]) / cfg.train.grad_accum
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip,
                                                       error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()

            if step % cfg.train.log_every == 0 or step + 1 == cfg.train.steps:
                record = {"step": step + 1, **{k: round(v, 4) for k, v in totals.items()},
                          "budget": budget, "residual_weight": round(residual_weight, 4),
                          "grad_norm": round(float(grad_norm), 3),
                          "lr": scheduler.get_last_lr()[0],
                          "seconds": round(time.time() - started, 1)}
                print(record, flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()

            done = step + 1
            if validation is not None and (done % cfg.train.eval_every == 0
                                           or done == cfg.train.steps):
                name, loader = validation
                aggregate, _ = evaluate(model, loader, device,
                                        cfg.train.gen_max_new_tokens,
                                        cfg.readout.max_budget)
                # evaluate() leaves the model in eval mode, which would silently
                # disable LoRA dropout for the rest of training.
                model.train()
                score = float(aggregate[cfg.train.select_metric])
                record = {"step": done, "split": name, "val_budget": cfg.readout.max_budget,
                          **{k: round(float(aggregate[k]), 4)
                             for k in ("em", "substring", "f1")},
                          "seconds": round(time.time() - started, 1)}
                print(f"[val] {record}", flush=True)
                log.write(json.dumps({"validation": record}) + "\n")
                log.flush()
                if score > best["metric"]:
                    best = {"metric": score, "step": done,
                            "metrics": {k: float(aggregate[k])
                                        for k in ("em", "substring", "f1")}}
                    model.save(os.path.join(cfg.train.out_dir, "checkpoint_best.pt"),
                               step=done)

    model.save(os.path.join(cfg.train.out_dir, "checkpoint_last.pt"),
               optimizer=optimizer, scheduler=scheduler, step=cfg.train.steps)
    if best["step"] is not None:
        # Reported, never silently substituted: the final evaluation below still
        # scores checkpoint_last, so "best" cannot be read as the headline number
        # unless a run explicitly asks for it.
        print(f"[val] best {cfg.train.select_metric}={best['metric']:.4f} "
              f"at step {best['step']} (checkpoint_best.pt)")
        with open(os.path.join(cfg.train.out_dir, "best_checkpoint.json"),
                  "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)
    run_evaluations(model, eval_loaders, device, cfg, args, cache)


if __name__ == "__main__":
    main()
