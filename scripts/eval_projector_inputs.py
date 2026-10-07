"""Frozen-checkpoint, paired attribution of projector input length/direction."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as config_module
from src import metrics
from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset, encode_text, move_to_device
from src.model import build_model
from src.projector_diagnostics import (MODES, DecoderTrace, ScalarDistributions, answer_scores,
    generation_record, geometry_values, input_variants, merge_trace, prompt_regions,
    question_positions, summarize_trace)
from src.prompt import assemble_inputs


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_config(payload):
    """Restore saved settings, rejecting unknown fields instead of ignoring them."""
    saved = payload.get("config")
    if not isinstance(saved, dict):
        raise ValueError("checkpoint must contain its full config")
    classes = {"readout": config_module.ReadoutConfig, "query_encoder": config_module.QueryEncoderConfig,
               "generator": config_module.GeneratorConfig, "decoder": config_module.DecoderInputConfig,
               "data": config_module.DataConfig, "train": config_module.TrainConfig}
    if set(saved) != set(classes):
        raise ValueError("checkpoint config sections do not match this code version")
    sections = {}
    for name, cls in classes.items():
        unknown = set(saved[name]) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown checkpoint config fields in {name}: {sorted(unknown)}")
        sections[name] = cls(**saved[name])
    cfg = config_module.Config(**sections)
    cfg.revalidate()
    if cfg.readout.kind not in {"joint_projector", "shared_projector"}:
        raise ValueError("requires an aligned full-budget projector checkpoint")
    if payload.get("generator_trainable") or cfg.generator.lora_init != "frozen":
        raise ValueError("this evaluation requires a frozen reader checkpoint")
    return cfg


def full_answer_ids(tokenizer, answer):
    eos = tokenizer.eos_token_id
    ids = encode_text(tokenizer, " "+answer.strip())
    if eos is None or not ids or eos in ids:
        raise ValueError("need a nonempty answer without embedded EOS and a tokenizer EOS")
    return ids+[eos]


@torch.no_grad()
def evaluate_batch(model, batch, max_new_tokens, geometry, traces, trace_decoder=False,
                   wrong_answers=None, eps=1e-8):
    """Read out E once; all arms share that E, the document mask and prompts."""
    model.eval()
    result = model.readout_cached(batch)
    e, mask = result["soft_tokens"], result["soft_token_mask"].bool()
    z = batch["cached_latents"].flatten(1, 2)
    expected_mask = batch["document_mask"].bool().repeat_interleave(
        batch["cached_latents"].size(2), dim=1)
    if z.shape != e.shape or not torch.equal(mask, expected_mask):
        raise ValueError("original and projected memory must have identical document/token slots")
    variants, degenerate = input_variants(z, e, mask, eps)
    prompts = model.build_prompts(batch, mask, training=False)
    builder = model.prompt_builders["D0"]
    empty_prompts = [builder.build("", int(m.sum())) for m in mask]
    spans = [question_positions(p, q) for p, q in zip(prompts, empty_prompts)]
    embedding = model.lm.get_input_embeddings()
    values, _ = geometry_values(z, e, mask, eps)
    geometry.add(values)
    # Actual IDs in the decoder prompt, including tokenizer boundary changes.
    word_values = []
    for prompt, positions in zip(prompts, spans):
        if positions:
            ids = torch.tensor([prompt.input_ids[p] for p in positions], device=e.device)
            word_values.append(embedding(ids).float().square().mean(-1).sqrt())
    geometry.add({"decoder_question_word_rms": torch.cat(word_values) if word_values else e.new_empty(0)})
    # No gold answer truncation: EOS is measured after the complete chosen alias.
    targets = [full_answer_ids(model.tok, row["target"]) for row in batch["raw"]]
    rows = []
    for i, raw in enumerate(batch["raw"]):
        row_values, row_counts = geometry_values(z[i:i+1], e[i:i+1], mask[i:i+1], eps)
        rows.append({"id": str(batch["ids"][i]), "query": batch["queries"][i],
                     "readout_query": batch["readout_queries"][i],
                     "golds": raw["answers"], "target": raw["target"],
                     "doc_ids": batch["retrieved_doc_ids"][i],
                     "memory_tokens": int(mask[i].sum()),
                     "prompt_input_ids": prompts[i].input_ids,
                     "memory_slot_positions": prompts[i].slot_positions,
                     "question_span_positions": spans[i],
                     "readout_query_truncated": batch["query_truncated"][i],
                     "training_target_would_truncate": len(targets[i])-1 > model.cfg.data.max_answer_len,
                     "geometry": {key: float(v.mean()) if v.numel() else None for key, v in row_values.items()},
                     "degenerate": row_counts,
                     "strata": {key: raw[key] for key in ("type", "hop_type", "level") if key in raw},
                     "modes": {}})
    for mode, memory in variants.items():
        actual = memory[mask].to(embedding.weight.dtype).float()
        geometry.add({mode+"_decoder_input_rms": actual.square().mean(-1).sqrt()})
        packed = assemble_inputs(embedding, prompts, memory, mask, targets, model.pad_id)
        trace = DecoderTrace(model.lm, prompt_regions(prompts, spans, packed["labels"])) if trace_decoder else None
        with trace if trace else nullcontext():
            output = model.lm(**packed)
            scores = answer_scores(output.logits, packed["labels"], model.tok.eos_token_id)
        if trace:
            merge_trace(traces.setdefault(mode, {}), trace.stats)
        del output, packed
        gen_packed = assemble_inputs(embedding, prompts, memory, mask, pad_token_id=model.pad_id, pad_side="left")
        ids = model.lm.generate(inputs_embeds=gen_packed["inputs_embeds"],
                               attention_mask=gen_packed["attention_mask"],
                               max_new_tokens=max_new_tokens, do_sample=False,
                               eos_token_id=model.tok.eos_token_id, pad_token_id=model.tok.pad_token_id)
        if ids.size(0) != len(rows):
            raise ValueError("generation returned a different batch size")
        for i, row in enumerate(rows):
            generated = generation_record(ids[i].tolist(), model.tok, max_new_tokens)
            row["modes"][mode] = {**generated, **metrics.score(generated["prediction"], row["golds"]),
                                   **scores[i]}
            if wrong_answers is not None:
                wrong = wrong_answers[row["id"]]
                wt = full_answer_ids(model.tok, wrong)
                candidate_packed = assemble_inputs(embedding, [prompts[i]], memory[i:i+1], mask[i:i+1], [wt], model.pad_id)
                candidate_output = model.lm(**candidate_packed)
                ws = answer_scores(candidate_output.logits, candidate_packed["labels"], model.tok.eos_token_id)[0]
                row["modes"][mode].update({"wrong_answer": wrong,
                    "wrong_content_tokens": ws["content_tokens"],
                    "wrong_content_logprob_mean": ws["content_logprob_mean"],
                    "content_candidate_margin": scores[i]["content_logprob_mean"]-ws["content_logprob_mean"]})
                del candidate_output, candidate_packed
        del gen_packed, ids
    return rows, degenerate


def qa_summary(rows):
    names = ("em", "f1", "substring", "content_logprob_mean", "eos_after_gold_probability",
             "eos_after_gold_logprob", "eos_during_gold_mean_probability", "generated_content_tokens",
             "eos_reached", "hit_generation_cap")
    if "content_candidate_margin" in rows[0]["modes"]["full"]:
        names += ("content_candidate_margin",)
    result = {}
    for mode in MODES:
        result[mode] = {name: sum(row["modes"][mode][name] for row in rows)/len(rows) for name in names}
        nt = sum(row["modes"][mode]["content_tokens"] for row in rows)
        result[mode]["token_weighted_content_logprob"] = sum(
            row["modes"][mode]["content_logprob_sum"] for row in rows)/nt
        result[mode]["n"] = len(rows)
    return result


def load_candidates(path, dataset):
    with open(path, encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    candidates = {}
    for row in records:
        key = str(row["id"])
        if key in candidates or not isinstance(row.get("wrong_answer"), str) or not row["wrong_answer"].strip():
            raise ValueError("candidate file requires unique IDs and nonempty wrong_answer strings")
        candidates[key] = row["wrong_answer"]
    if set(candidates) != {str(row["id"]) for row in dataset.rows}:
        raise ValueError("candidate IDs must exactly match the evaluated subset")
    for row in dataset.rows:
        wrong = metrics.normalize_answer(candidates[str(row["id"])])
        if not wrong or wrong in {metrics.normalize_answer(gold) for gold in row["answers"]}:
            raise ValueError(f"wrong answer overlaps a gold alias: {row['id']}")
    return candidates


def build_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=None, help="default: all rows; positive values select a fixed prefix")
    parser.add_argument("--max_new_tokens", type=int, default=None, help="default: checkpoint generation cap")
    parser.add_argument("--cache_dir", default=None, help="relocate the same cache")
    parser.add_argument("--generator_path", default=None, help="relocate the same frozen reader")
    parser.add_argument("--candidates_file", default=None, help="JSONL: one curated wrong_answer per evaluated id")
    parser.add_argument("--trace_decoder", action="store_true", help="scalar residual/update hooks in gold scoring only")
    parser.add_argument("--norm_eps", type=float, default=1e-8)
    return parser


def main(argv=None):
    args = build_args().parse_args(argv)
    if (args.batch_size < 1 or args.max_samples is not None and args.max_samples < 1
            or args.max_new_tokens is not None and args.max_new_tokens < 1
            or not 0 < args.norm_eps < float("inf")):
        raise ValueError("batch/sample/generation limits and norm_eps must be positive")
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("use a new/empty out_dir; attribution runs are never silently overwritten")
    # Existing trusted project checkpoints use the same loader in QuROModel.load.
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint_config(payload)
    saved_config = asdict(cfg)
    del payload
    cfg.train.out_dir = str(out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    cfg.train.device, cfg.generator.device = str(device), str(device)
    if args.cache_dir:
        cfg.data.cache_dir = args.cache_dir
    if args.generator_path:
        cfg.generator.name_or_path = args.generator_path
    cap = args.max_new_tokens if args.max_new_tokens is not None else cfg.train.gen_max_new_tokens
    if cap < 1:
        raise ValueError("checkpoint generation cap must be positive")
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.train.seed)
    cache = LatentCache(cfg.data.cache_dir)
    if cfg.generator.n_mem_tokens is not None and cfg.generator.n_mem_tokens != cache.metadata.latent_size:
        raise ValueError("checkpoint memory slots and cache disagree")
    stack, model = build_model(cfg, cache.metadata.hidden_size)
    model.to(device)
    model.lm.to(device)
    missing, unexpected, step = model.load(args.checkpoint)
    if unexpected or any(name.startswith("readout.") for name in missing):
        raise ValueError(f"checkpoint did not restore the complete projector: {missing}, {unexpected}")
    if any(parameter.requires_grad for parameter in model.lm.parameters()):
        raise ValueError("reader unexpectedly has trainable parameters")
    model.eval()
    dataset = QuRODataset(args.eval_file, stack.tokenizer, cfg.data, stack.query_tokenizer, limit=args.max_samples)
    if not dataset.rows or len({str(row["id"]) for row in dataset.rows}) != len(dataset):
        raise ValueError("evaluation data must have nonempty, unique IDs")
    candidates = load_candidates(args.candidates_file, dataset) if args.candidates_file else None
    collator = QuROCollator(cache, model.pad_id, getattr(stack.query_tokenizer, "pad_token_id", model.pad_id),
                           max_docs=cfg.data.max_docs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    geometry, traces, rows, counts = ScalarDistributions(), {}, [], {}
    out_dir.mkdir(parents=True, exist_ok=True)
    # Flush each completed batch: a long eval can be inspected while it runs.
    with open(out_dir/"predictions.jsonl", "w", encoding="utf-8") as handle:
        for batch in loader:
            new_rows, degenerate = evaluate_batch(model, move_to_device(batch, device), cap,
                geometry, traces, args.trace_decoder, candidates, args.norm_eps)
            for row in new_rows:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
            handle.flush()
            rows.extend(new_rows)
            for name, value in degenerate.items():
                counts[name] = counts.get(name, 0)+value
            print(f"evaluated {len(rows)}/{len(dataset)}", flush=True)
    manifest_digest = hashlib.sha256(json.dumps(cache.manifest, sort_keys=True).encode()).hexdigest()
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    report = {"schema_version": 1, "completed": True, "modes": list(MODES), "units": "fraction",
              "checkpoint": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": file_digest(args.checkpoint),
              "checkpoint_step": step, "checkpoint_config": saved_config, "effective_config": asdict(cfg),
              "eval_file": str(Path(args.eval_file).resolve()), "eval_file_sha256": file_digest(args.eval_file),
              "evaluated_ids_sha256": hashlib.sha256(json.dumps([r['id'] for r in rows]).encode()).hexdigest(),
              "cache_manifest_sha256": manifest_digest, "cache_metadata": asdict(cache.metadata),
              "generator_path": cfg.generator.name_or_path, "git_commit": git_commit,
              "torch_version": torch.__version__, "device": str(device), "batch_size": args.batch_size,
              "max_samples": args.max_samples, "max_new_tokens": cap,
              "generation_cap_overridden": cap != saved_config["train"]["gen_max_new_tokens"],
              "norm_eps": args.norm_eps, "degenerate": counts,
              "candidates_sha256": file_digest(args.candidates_file) if args.candidates_file else None,
              "teacher_forcing": "full first gold alias; EOS excluded from content; no answer truncation",
              "word_embedding_reference": "actual D0 question insertion span; includes boundary-token changes",
              "trace_context": "gold teacher-forced forward only" if args.trace_decoder else None,
              "toy_smoke_only": cfg.generator.kind == "toy",
              "qa": qa_summary(rows), "geometry": geometry.summary(),
              "decoder_trace": {mode: summarize_trace(totals) for mode, totals in traces.items()}}
    with open(out_dir/"result.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(report["qa"], ensure_ascii=False, indent=2), flush=True)
    return report


if __name__ == "__main__":
    main()
