"""Plan/launch the fixed-K evidence evaluation and equal-update data-scale grid.

No torch import is needed to inspect a plan. --execute runs ordinary src.train
processes, at most one per listed GPU, with checked exit codes and separate logs.
Run this launcher inside tmux for long server jobs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import prepare_public_qa as public
from prepare_open_rag import named_paths

REPO = Path(__file__).resolve().parents[1]


def cache_manifest(args):
    path = Path(args.cache_dir) / "manifest.json"
    manifest = json.loads(path.read_text())
    expected = {"latent_size": 8, "hidden_size": 4096, "doc_max_length": 128, "compr_rate": 16}
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"cache must use published PISCO r16: {name}={manifest.get(name)}, expected {value}")
    if manifest.get("checkpoint") and Path(manifest["checkpoint"]).resolve() != Path(args.generator_path).resolve():
        raise ValueError("cache compressor checkpoint differs from the requested frozen reader")
    if not Path(args.generator_path).is_dir():
        raise ValueError(f"missing generator directory: {args.generator_path}")
    return manifest


def checked_rows(path, cache, k):
    rows = list(public.read_jsonl(path))
    if not rows:
        raise ValueError(f"empty file: {path}")
    seen = set()
    for row in rows:
        identifier = row.get("id")
        docs = row.get("retrieved_doc_ids", [])
        if not identifier or identifier in seen or not public.question(row) or not row.get("answers"):
            raise ValueError(f"invalid/duplicate labelled question in {path}: {identifier}")
        if len(docs) != k or len(set(docs)) != k or any(d not in cache["documents"] for d in docs):
            raise ValueError(f"{path}: {identifier} needs exactly {k} distinct, cached documents")
        seen.add(identifier)
    return rows


def common_command(args, train_file, evals, k):
    return [sys.executable, "-m", "src.train", "--preset", "pisco_shared_projector",
            "--train_file", str(train_file), "--eval_files", evals,
            "--cache_dir", str(Path(args.cache_dir).resolve()),
            "--generator_path", str(Path(args.generator_path).resolve()),
            "--max_docs", str(k), "--max_query_len", "256", "--max_answer_len", "128",
            "--gen_max_new_tokens", "128", "--projector_fusion", "none",
            "--support_loss_weight", "0", "--seed", str(args.seed)]


def training_plan(args):
    cache = cache_manifest(args)
    full_root, small_root = Path(args.full_ready).resolve(), Path(args.small_ready).resolve()
    full_path, small_path = full_root / "train.jsonl", small_root / "train.jsonl"
    full = checked_rows(full_path, cache, 5)
    small = checked_rows(small_path, cache, 5)
    dev_path = full_root / "dev.jsonl"
    dev = checked_rows(dev_path, cache, 5)
    if public.digest_file(dev_path) != public.digest_file(small_root / "dev.jsonl"):
        raise ValueError("full/small cohorts must use byte-identical dev data/order")
    indexed = {r["id"]: r for r in full}
    for row in small:
        large = indexed.get(row["id"])
        if large is None or any(row.get(key) != large.get(key) for key in ("query", "answers", "retrieved_doc_ids", "source")):
            raise ValueError("small cohort must be an identical labelled/evidence subset of full")
    train_queries = {public.normalized_question(r["query"]) for r in full}
    if train_queries & {public.normalized_question(r["query"]) for r in dev}:
        raise ValueError("train/dev question overlap")
    if args.batch_size < 1 or args.grad_accum < 1 or len(small) < args.batch_size:
        raise ValueError("invalid effective batch or too few training rows")
    # Match DataLoader(drop_last=True), allowing the last accumulation to
    # cross into the next shuffle. Report this as approximately one full pass.
    steps = args.steps if args.steps is not None else math.ceil((len(full) // args.batch_size) / args.grad_accum)
    if steps < 1:
        raise ValueError("steps must be positive")
    effective_batch = args.batch_size * args.grad_accum
    snapshots = sorted({s for s in (9000, math.ceil(3 * (len(small) // args.batch_size) / args.grad_accum), steps)
                        if 0 < s <= steps})
    jobs = []
    for cohort, path in (("full", full_path), ("90k", small_path)):
        if cohort not in getattr(args, "cohorts", ("full", "90k")):
            continue
        for arm, mode in (("SQ", "conditioned"), ("S0", "none")):
            name = f"{cohort}_{arm}_s{args.seed}"
            command = common_command(args, path, f"dev={dev_path}", 5)
            command += ["--steps", str(steps), "--lr", "5e-5", "--batch_size", str(args.batch_size),
                        "--grad_accum", str(args.grad_accum), "--data_order_seed", str(args.data_order_seed),
                        "--eval_every", "1000", "--eval_every_samples", "1000",
                        "--eval_max_samples", str(len(dev)), "--select_metric", "f1", "--num_workers", "4",
                        "--checkpoint_steps", ",".join(map(str, snapshots)),
                        "--projector_query_mode", mode, "--tag", name,
                        "--out_dir", str(Path(args.out_dir).resolve() / name)]
            jobs.append({"name": name, "command": command})
    return {"experiment": "equal_updates_data_scale", "steps": steps,
            "effective_batch": effective_batch, "sample_presentations_nominal": steps * effective_batch,
            "train_rows": {"full": len(full), "90k": len(small)},
            "unique_normalized_questions": {"full": len(train_queries),
                                             "90k": len({public.normalized_question(r['query']) for r in small})},
            "train_by_source": {"full": dict(Counter(r.get('source', 'unknown') for r in full)),
                                "90k": dict(Counter(r.get('source', 'unknown') for r in small))},
            "nominal_passes": {"full": steps * effective_batch / len(full),
                               "90k": steps * effective_batch / len(small)},
            "dev_selection": "first_1000_fixed_mixed_dev_f1", "snapshots": snapshots,
            "inputs": public.file_records([full_path, small_path, dev_path, Path(args.cache_dir) / "manifest.json"]),
            "jobs": jobs}


def evaluation_plan(args):
    if args.max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    cache = cache_manifest(args)
    evals = named_paths(args.eval_files)
    maximum = 0
    for path in evals.values():
        maximum = max(maximum, len(checked_rows(path, cache, args.k)))
    checkpoints = named_paths(args.checkpoints)
    modes = {"SQ": "conditioned", "S0": "none", "old_SQ": "conditioned", "old_S0": "none",
             "SQX": "conditioned", "S0X": "none", "full_SQ": "conditioned", "full_S0": "none",
             "90k_SQ": "conditioned", "90k_S0": "none"}
    if set(checkpoints) - modes.keys():
        raise ValueError(f"checkpoint names must be one of {sorted(modes)}")
    jobs = []
    baselines = [] if getattr(args, "skip_pisco", False) else [("PISCO", None)]
    if not baselines and not checkpoints:
        raise ValueError("no evaluation jobs requested")
    for name, checkpoint in [*baselines, *checkpoints.items()]:
        command = common_command(args, next(iter(evals.values())).resolve(),
                                 ",".join(f"{n}={p.resolve()}" for n, p in evals.items()), args.k)
        command += ["--eval_only", "--eval_max_samples", str(args.max_samples or maximum),
                    "--tag", name, "--out_dir", str(Path(args.out_dir).resolve() / name)]
        if checkpoint is None:
            budget = args.k * 8
            command += ["--readout", "pisco_direct", "--budget", str(budget),
                        "--budget_buckets", str(budget), "--no_budget_dropout"]
        else:
            if not checkpoint.is_file():
                raise ValueError(f"missing checkpoint: {checkpoint}")
            command += ["--resume_from", str(checkpoint.resolve()), "--projector_query_mode", modes[name]]
            if name.endswith("X"):
                command += ["--projector_cross_document"]
        jobs.append({"name": name, "command": command})
    return {"experiment": f"common_retrieved_k{args.k}_evaluation", "max_docs": args.k,
            "max_samples": args.max_samples or maximum,
            "inputs": public.file_records([*evals.values(), *checkpoints.values(), Path(args.cache_dir) / "manifest.json"]),
            "jobs": jobs}


def launch(args, plan):
    gpus = args.gpus.split(",")
    if not gpus or any(not re_gpu(g) for g in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError("--gpus must be distinct numeric GPU IDs, e.g. 0,1,2,3")
    out = Path(args.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "execution_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("existing output has a different plan/input hash; use a new directory")
    for job in plan["jobs"]:
        directory = out / job["name"]
        if directory.exists() and any(directory.iterdir()):
            raise ValueError(f"run output is not empty: {directory}; use a new directory")
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    for job in plan["jobs"]:
        print(shlex.join(job["command"]), flush=True)
    if not args.execute:
        print(f"Plan saved to {plan_path}. Pass --execute to run it.")
        return
    waiting, active, failures = list(plan["jobs"]), {}, []
    try:
        while waiting or active:
            for gpu in gpus:
                if waiting and gpu not in active:
                    job = waiting.pop(0)
                    (out / job["name"]).mkdir()
                    handle = (out / f"{job['name']}.log").open("w")
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
                    process = subprocess.Popen(job["command"], cwd=REPO, env=env,
                                               stdout=handle, stderr=subprocess.STDOUT)
                    active[gpu] = (process, handle, job["name"])
                    print(f"GPU {gpu}: launched {job['name']}", flush=True)
            for gpu, (process, handle, name) in list(active.items()):
                code = process.poll()
                if code is not None:
                    handle.close()
                    del active[gpu]
                    print(f"GPU {gpu}: {name} exited {code}", flush=True)
                    if code:
                        failures.append((name, code))
                        waiting.clear()
            if active:
                time.sleep(1)
    finally:
        for process, handle, _ in active.values():
            process.terminate()
            process.wait()
            handle.close()
    if failures:
        raise SystemExit(f"failed jobs: {failures}; see logs under {out}")


def re_gpu(value):
    return value.isdigit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--cache_dir", required=True)
        p.add_argument("--generator_path", default="/data02/quro/models/pisco-mistral")
        p.add_argument("--out_dir", required=True)
        p.add_argument("--gpus", default="0,1,2,3")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--execute", action="store_true", help="otherwise only validate/save/print the plan")
    p = sub.choices["train"]
    p.add_argument("--full_ready", required=True)
    p.add_argument("--small_ready", required=True)
    p.add_argument("--steps", type=int, help="default: approximately one full-pool pass for BOTH cohorts")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--data_order_seed", type=int, default=42)
    p.add_argument("--cohorts", nargs="+", choices=["full", "90k"], default=["full", "90k"],
                   help="default: both equal-update cohorts; use full for the later seed replication")
    p = sub.choices["eval"]
    p.add_argument("--eval_files", nargs="+", required=True, help="split_name=/path.jsonl")
    p.add_argument("--checkpoints", nargs="*", default=[], help="SQ=/path.pt S0=/path.pt old_SQ=/path.pt")
    p.add_argument("--k", type=int, choices=[5, 10], default=5)
    p.add_argument("--max_samples", type=int, default=0, help="0 evaluates all rows")
    p.add_argument("--skip_pisco", action="store_true", help="reuse an already evaluated published baseline on the same data")
    args = parser.parse_args()
    launch(args, training_plan(args) if args.command == "train" else evaluation_plan(args))


if __name__ == "__main__":
    main()
