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
    ap.add_argument("--preset", default="pisco_smoke", choices=["toy", "pisco_smoke", "pisco_gonogo"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--resume_from", default=None)

    ap.add_argument("--readout", choices=["quro", "pisco_direct", "similarity_topb"], default=None)
    ap.add_argument("--output_query_mode",
                    choices=["agnostic", "agnostic_matched", "add", "film", "concat", "xattn"],
                    default=None)
    # Sets output_query_mode and cosine_prior together, so an arm cannot be
    # half-specified the way the historical A runs were (warning_and_target W1).
    ap.add_argument("--arm", choices=["A0", "A1", "C0", "C1", "S", "P"], default=None)
    ap.add_argument("--agnostic_param_matched", action="store_true",
                    help="A arms use agnostic_matched, so A and C have equal parameters")
    ap.add_argument("--budget", type=int, default=None, help="B_max and the fixed eval budget")
    ap.add_argument("--budget_buckets", default=None, help="comma-separated discrete B values")
    ap.add_argument("--no_budget_dropout", action="store_true")
    ap.add_argument("--readout_blocks", type=int, default=None)
    ap.add_argument("--d_readout", type=int, default=None)
    # Legacy: equivalent to --readout_output_mode delta_only (and, historically,
    # it also swapped out_proj to the default init -- pass --out_proj_init default
    # to reproduce that exactly).  See docs/warning_and_target.md W2.
    ap.add_argument("--no_residual_readout", action="store_true")
    ap.add_argument("--readout_output_mode",
                    choices=["full", "pool_only", "delta_only"], default=None)
    ap.add_argument("--out_proj_init", choices=["zeros", "default"], default=None)
    ap.add_argument("--no_cosine_prior", action="store_true")
    ap.add_argument("--prior_mode", choices=["rank", "shared"], default=None)
    ap.add_argument("--tau_init", type=float, default=None)
    ap.add_argument("--without_document_source", action="store_true")
    ap.add_argument("--adaptive_budget", action="store_true")

    ap.add_argument("--decoder_input_mode",
                    choices=["D0", "D1", "D2", "D3", "AG", "RG"], default=None)
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
        # (docs/warning_and_target.md W4).
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
        "readout_output_mode": cfg.readout.output_mode,
        "out_proj_init": cfg.readout.out_proj_init,
        "generator_lora_init": cfg.generator.lora_init,
        "train_decoder_input_mode": cfg.decoder.input_mode,
        "query_text_dropout": cfg.decoder.query_text_dropout,
        "offline_m": cache.metadata.latent_size,
        "offline_compressor": cache.metadata.compressor,
        "offline_compr_rate": cache.metadata.compr_rate,
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
    cfg = apply_overrides(get_config(args.preset), args)
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
    cfg.to_json(os.path.join(cfg.train.out_dir, "config.json"))
    print(cfg.summary())
    print(f"[params] {json.dumps({k: round(v/1e6, 2) for k, v in model.parameter_report().items()})} (M)")

    collator = QuROCollator(
        cache, pad_id=model.pad_id,
        query_pad_id=getattr(stack.query_tokenizer, "pad_token_id", model.pad_id),
        max_docs=cfg.data.max_docs,
        require_budget_labels=cfg.readout.adaptive_budget and not args.eval_only)
    train_set, train_loader, eval_loaders = build_loaders(
        cfg, stack.tokenizer, stack.query_tokenizer, collator, args.query_control,
        args.doc_control, corpus)
    print(f"[data] train={len(train_set)} device={device}")

    if args.eval_only:
        if cfg.train.resume_from:
            model.load(cfg.train.resume_from)
        run_evaluations(model, eval_loaders, device, cfg, args, cache)
        return

    params = model.trainable_parameters()
    if not params:
        raise RuntimeError("nothing is trainable; check readout kind and generator_lora_init")
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(cfg.train.steps, cfg.train.warmup_ratio))
    start_step = 0
    if cfg.train.resume_from:
        _, _, saved = model.load(cfg.train.resume_from, optimizer=optimizer, scheduler=scheduler)
        start_step = int(saved or 0)

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
            totals = {"loss": 0.0, "qa_loss": 0.0, "mean_budget": 0.0, "residual_penalty": 0.0}
            for _ in range(max(1, cfg.train.grad_accum)):
                batch = move_to_device(next(iterator), device)
                output = model(batch, budget=budget, residual_weight=residual_weight)
                (output["loss"] / cfg.train.grad_accum).backward()
                for key in totals:
                    if key in output:
                        totals[key] += float(output[key]) / cfg.train.grad_accum
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
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

    model.save(os.path.join(cfg.train.out_dir, "checkpoint_last.pt"),
               optimizer=optimizer, scheduler=scheduler, step=cfg.train.steps)
    run_evaluations(model, eval_loaders, device, cfg, args, cache)


if __name__ == "__main__":
    main()
