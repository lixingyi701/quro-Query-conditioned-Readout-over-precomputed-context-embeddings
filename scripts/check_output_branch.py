"""Is the readout *selecting* evidence, or *synthesising* it?

The output is ``E = s * AttnPool(alpha, Z) + Delta``: a pooling branch that
returns a convex mixture of the cached latents themselves, and a free branch that
can emit anything.  Everything the method claims rests on the first one.  If the
answer is carried by ``Delta``, then ``alpha`` is not the causal path, "readout"
is the wrong name, and -- concretely -- supervising the attention towards the
gold paragraphs would train a branch that does not drive the output.

This is why the question comes before the choice of auxiliary loss rather than
after it.

The intervention is at inference: one trained checkpoint, evaluated three times
with only the output composition changed.  That measures what the trained model
*currently relies on*.  It does not measure what it could compensate for if
retrained under a branch -- that needs a retrain, and the two answer different
questions (HANDOFF.md §3 W2).

Reading the result:

``pool_only`` near ``full``    the output is selection; attention is the causal
                              path and is worth supervising.
``pool_only`` collapses,      the output is synthesised.  Auxiliary losses should
``delta_only`` near ``full``  target the output distribution (KL), not attention.
both degrade                  the two branches are jointly required and neither
                              alone is interpretable as the mechanism.

The mismatch-document control runs under every branch too: a branch that keeps
its score by falling back on the decoder's parametric memory is not keeping it by
reading evidence, and only the gap to that floor distinguishes the two.

    python scripts/check_output_branch.py --run /data02/quro/runs/bs32_C1 --budget 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, arm_label
from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset
from src.model import build_model
from src.readout import OUTPUT_MODES, QuroReadout
from src.train import evaluate


def config_from_run(run_dir: str) -> Config:
    """Rebuild the exact config a run was trained with.

    Rebuilding from a preset instead would silently substitute today's defaults
    for whatever the run actually used, which is how a diagnostic ends up
    describing a model that was never trained.
    """
    with open(os.path.join(run_dir, "config.json"), encoding="utf-8") as f:
        raw = json.load(f)
    cfg = Config()
    for section, values in raw.items():
        target = getattr(cfg, section, None)
        if target is None or not hasattr(target, "__dataclass_fields__"):
            continue
        for key, value in values.items():
            if hasattr(target, key):
                setattr(target, key, value)
    cfg.revalidate()
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory holding config.json "
                                                 "and checkpoint_last.pt")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--budget", type=int, default=None, help="defaults to the run's max_budget")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--modes", nargs="+", default=list(OUTPUT_MODES))
    ap.add_argument("--out", default=None, help="where to write the report json")
    args = ap.parse_args()

    cfg = config_from_run(args.run)
    budget = args.budget or cfg.readout.max_budget
    # The LoRA is restored from the checkpoint; re-initialising it from PISCO here
    # would quietly evaluate a different decoder than the one that was trained.
    cfg.generator.lora_init = "frozen"
    cfg.train.eval_max_samples = args.rows

    cache = LatentCache(cfg.data.cache_dir)
    cfg.readout.cache_hidden = cache.metadata.hidden_size
    stack, model = build_model(cfg, cache_hidden=cache.metadata.hidden_size)
    if not isinstance(model.readout, QuroReadout):
        raise SystemExit(f"{args.run} uses {type(model.readout).__name__}, which has no "
                         "pooling/Delta split; only the quro readout does")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    stack.lm.to(device)
    _, _, step = model.load(os.path.join(args.run, "checkpoint_last.pt"))
    model.eval()
    print(f"loaded {args.run} (step {step}, arm {arm_label(cfg)}, B={budget}, "
          f"trained output_mode={cfg.readout.output_mode})")

    collator = QuROCollator(
        cache, pad_id=model.pad_id,
        query_pad_id=getattr(stack.query_tokenizer, "pad_token_id", model.pad_id),
        max_docs=cfg.data.max_docs)
    path = cfg.data.resolved_eval_files()[args.split]
    loaders = {}
    for name, doc_shift in (("clean", 0), ("mismatch-doc", 1)):
        dataset = QuRODataset(path, stack.tokenizer, cfg.data,
                              query_tokenizer=stack.query_tokenizer,
                              document_shift=doc_shift, limit=args.rows)
        loaders[name] = DataLoader(dataset, batch_size=cfg.train.eval_batch_size,
                                   shuffle=False, collate_fn=collator)

    # evaluate() calls readout_cached() without an override, so the branch is
    # switched on the module for the duration of each pass rather than threaded
    # through as an argument.  Restored afterwards so a later pass is not affected.
    original = model.readout.output_mode
    report = {"run": os.path.basename(args.run), "step": step, "arm": arm_label(cfg),
              "budget": budget, "trained_output_mode": original, "modes": {}}
    try:
        for mode in args.modes:
            model.readout.output_mode = mode
            entry = {}
            for name, loader in loaders.items():
                aggregate, _ = evaluate(model, loader, device,
                                        cfg.train.gen_max_new_tokens, budget)
                entry[name] = {k: aggregate[k] for k in
                               ("em", "substring", "f1", "n", "constant_baseline_em")}
            entry["evidence_em"] = round(
                entry["clean"]["em"] - entry["mismatch-doc"]["em"], 6)
            report["modes"][mode] = entry
            print(f"  {mode:<11} EM={100*entry['clean']['em']:.2f}  "
                  f"sub={100*entry['clean']['substring']:.2f}  "
                  f"floor={100*entry['mismatch-doc']['em']:.2f}  "
                  f"evidence={100*entry['evidence_em']:.2f}", flush=True)
    finally:
        model.readout.output_mode = original

    out = args.out or os.path.join(args.run, "output_branch.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"wrote {out}")

    if "full" in report["modes"] and "pool_only" in report["modes"]:
        full = report["modes"]["full"]["clean"]["em"]
        pool = report["modes"]["pool_only"]["clean"]["em"]
        share = pool / full if full else float("nan")
        print(f"\npool_only keeps {100*share:.1f}% of full's EM"
              + ("  -> the output is selection; attention is worth supervising"
                 if share > 0.8 else
                 "  -> the output is largely synthesised; target the output "
                 "distribution, not attention"))


if __name__ == "__main__":
    main()
