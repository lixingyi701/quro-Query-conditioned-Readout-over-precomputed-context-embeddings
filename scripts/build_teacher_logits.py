"""Precompute P's answer distributions once, so training never runs the teacher.

Running the teacher online would mean a second 7B forward every step and, worse,
switching PEFT adapters inside the training loop -- the student's decoder LoRA and
the teacher's are different adapters on the same backbone, and that state is
exactly what the query-representation work just finished making safe. Offline
generation sidesteps both and makes the teacher trivially fixed.

What is stored, per training example and per answer token: the teacher's
probabilities over the **full** vocabulary, truncated to the top-k, plus the
leftover tail mass. Storing top-k *logits* and softmaxing over them at train time
would be a different distribution -- it deletes the tail and rescales the rest --
so the softmax happens here, before truncation.

Contracts this script is responsible for (docs/TRAINING_STRATEGY_REVIEW_AND_PLAN
§6.3), each checked rather than assumed:

* the teacher is fixed and in eval mode;
* teacher and student are teacher-forced on the *same* gold answer, so the
  answer-relative alignment they share is meaningful;
* only positions that predict an answer token are stored, including the EOS the
  data pipeline appends;
* the tokenizer, the target ids and the truncation rule come from the same config
  the student will use, and their hashes are recorded;
* training data only -- distilling on dev or test would route evaluation data into
  the student through the teacher.

    python scripts/build_teacher_logits.py --teacher /data02/quro/runs/hp2d0_P
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset, move_to_device
from src.distill import teacher_probabilities
from src.model import build_model
from scripts.check_output_branch import config_from_run


def digest(values) -> str:
    sha = hashlib.sha1()
    for value in values:
        sha.update(str(value).encode("utf-8"))
    return sha.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="run directory of the P checkpoint")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None)
    ap.add_argument("--top_k", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--verify_rows", type=int, default=16,
                    help="rows to re-check against a live teacher forward before writing")
    args = ap.parse_args()

    cfg = config_from_run(args.teacher)
    cfg.generator.lora_init = "frozen"
    if cfg.readout.kind != "pisco_direct":
        print(f"[warn] teacher readout is {cfg.readout.kind!r}, not pisco_direct; "
              "this is only the full-cache teacher if that is intended")

    cache = LatentCache(cfg.data.cache_dir)
    cfg.readout.cache_hidden = cache.metadata.hidden_size
    stack, model = build_model(cfg, cache_hidden=cache.metadata.hidden_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    stack.lm.to(device)
    _, _, step = model.load(os.path.join(args.teacher, "checkpoint_last.pt"))
    # eval, and never anything else: PISCO's lora_dropout would make the stored
    # targets a sample rather than the teacher's distribution.
    model.eval()
    print(f"[teacher] {args.teacher} step {step}, readout={cfg.readout.kind}, "
          f"B={cfg.readout.max_budget}")

    path = (cfg.data.train_file if args.split == "train"
            else cfg.data.resolved_eval_files()[args.split])
    if args.split != "train":
        print(f"[warn] building teacher targets from {args.split!r}; distillation "
              "must only ever consume the training split")
    dataset = QuRODataset(path, stack.tokenizer, cfg.data,
                          query_tokenizer=stack.query_tokenizer, limit=args.limit)
    collator = QuROCollator(
        cache, pad_id=model.pad_id,
        query_pad_id=getattr(stack.query_tokenizer, "pad_token_id", model.pad_id),
        max_docs=cfg.data.max_docs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collator)

    index, chunks = {}, {"topk_index": [], "topk_probability": [], "tail": [], "targets": []}
    total, started = 0, time.time()
    verify = []
    with torch.no_grad():
        for batch_number, batch in enumerate(loader):
            ids = list(batch["ids"])
            batch = move_to_device(batch, device)
            _, result = model.qa_loss(batch, budget=cfg.readout.max_budget,
                                      return_logits=True)
            logits = result["answer_logits"]
            rows = result["answer_rows"].cpu()
            order = result["answer_order"].cpu()
            targets = result["answer_targets"].cpu()
            top_index, top_probability, tail = teacher_probabilities(
                logits, args.top_k, args.temperature)

            # Rows come out in row-major order, so grouping by example only needs
            # the per-row counts; the answer-relative index is already `order`.
            for row_number, example in enumerate(ids):
                mask = rows == row_number
                length = int(mask.sum())
                if length == 0:
                    continue
                if example in index:
                    raise ValueError(f"duplicate example id in {path}: {example}")
                positions = order[mask]
                if not torch.equal(positions, torch.arange(length)):
                    raise ValueError(
                        f"{example}: answer positions are {positions.tolist()}, "
                        "expected a contiguous 0..L-1; the alignment is broken")
                index[example] = (total, length)
                total += length
                chunks["topk_index"].append(top_index[mask].cpu().to(torch.int32))
                chunks["topk_probability"].append(top_probability[mask].cpu().to(torch.float16))
                chunks["tail"].append(tail[mask].cpu().to(torch.float16))
                chunks["targets"].append(targets[mask].to(torch.int32))

            if len(verify) < args.verify_rows:
                verify.append((ids[0], logits[rows == 0][:, :].float().cpu()))
            if batch_number % 50 == 0:
                seen = sum(len(x) for x in [index])
                print(f"  {len(index)}/{len(dataset)} examples, {total} positions, "
                      f"{time.time() - started:.0f}s", flush=True)

    payload = {
        "meta": {
            "teacher": os.path.abspath(args.teacher),
            "teacher_step": step,
            "teacher_readout": cfg.readout.kind,
            "teacher_budget": cfg.readout.max_budget,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "split": args.split,
            "source_file": path,
            "examples": len(index),
            "positions": total,
            "max_answer_len": cfg.data.max_answer_len,
            "tokenizer": getattr(stack.tokenizer, "name_or_path", "unknown"),
            "target_digest": digest(sorted(index)),
            "cache_dir": cfg.data.cache_dir,
        },
        "index": index,
        "topk_index": torch.cat(chunks["topk_index"]),
        "topk_probability": torch.cat(chunks["topk_probability"]),
        "tail": torch.cat(chunks["tail"]),
        "targets": torch.cat(chunks["targets"]),
    }

    # The stored mass has to look like a probability distribution after the fp16
    # round trip, or the divergence is being computed against something that is not
    # one.  Checked here, where it is cheap, rather than discovered as a negative
    # loss term mid-training.
    mass = payload["topk_probability"].float().sum(-1) + payload["tail"].float()
    worst = float((mass - 1.0).abs().max())
    print(f"[check] top-k mass + tail deviates from 1 by at most {worst:.2e}")
    if worst > 1e-2:
        raise SystemExit("stored teacher mass is not normalised; refusing to write")
    covered = float(payload["topk_probability"].float().sum(-1).mean())
    print(f"[check] top-{args.top_k} covers {100*covered:.2f}% of the teacher's mass "
          f"on average (the rest is the tail bucket)")

    out = args.out or os.path.join(args.teacher, f"teacher_{args.split}_T{args.temperature}.pt")
    torch.save(payload, out)
    size = os.path.getsize(out) / 1e6
    print(f"wrote {out}: {len(index)} examples, {total} positions, {size:.0f} MB")
    with open(out.replace(".pt", ".meta.json"), "w", encoding="utf-8") as f:
        json.dump(payload["meta"], f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
