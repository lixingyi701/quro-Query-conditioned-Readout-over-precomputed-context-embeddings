"""Degenerate QuRO into PISCO and check it reproduces PISCO's own output.

This is the strongest correctness test in the pipeline.  With
``readout=pisco_direct`` and no training, QuRO should be doing exactly what PISCO
does: take the cached latents, drop them into the memory slots of PISCO's prompt,
and decode.  If QuRO's answers match ``generate_from_text()`` on the same
documents, then the cache round-trip, the prompt template, the slot indexing and
the embedding injection are all correct simultaneously.

A mismatch localises the bug:
  * PISCO answers well, QuRO does not  -> prompt or injection is wrong;
  * both are wrong                     -> the cache or the compressor is wrong.

Do not expect 100% agreement, and do not chase the gap.  PISCO's compression is
deterministic for a fixed batch but *not* across batch sizes: measured on this
setup, the same document compressed in a batch of 16 versus 64 differs by up to
0.66 in absolute value (mean 0.024) while staying at cosine similarity >= 0.9997.
Greedy decoding over a 7B bf16 model amplifies that into an occasional synonym
swap after a dozen shared tokens.

Two consequences carry into the experiments:
  * every latent cache carries a batch-composition fingerprint, so the PISCO
    baseline must read the *same cache* rather than recompress on the fly --
    otherwise the comparison picks up an uncontrolled nuisance variable;
  * single-example comparisons between systems are noise; only aggregate metrics
    mean anything.  Agreement in the 70-90% range with matching aggregate scores
    is a pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import get_config
from src import metrics, paths
from src.cache import LatentCache
from src.data import QuROCollator, QuRODataset, move_to_device
from src.model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="pisco_smoke")
    ap.add_argument("--corpus", default=os.path.join(paths.DATA_DIR, "smoke/corpus.jsonl"))
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = get_config(args.preset)
    cfg.readout.kind = "pisco_direct"
    cfg.generator.lora_init = "frozen"
    cfg.revalidate()

    cache = LatentCache(cfg.data.cache_dir)
    cfg.readout.cache_hidden = cache.metadata.hidden_size
    stack, model = build_model(cfg, cache_hidden=cache.metadata.hidden_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    stack.lm.to(device)

    dataset = QuRODataset(cfg.data.eval_files["dev"], stack.tokenizer, cfg.data,
                          query_tokenizer=stack.query_tokenizer, limit=args.rows)
    collator = QuROCollator(cache, pad_id=model.pad_id,
                            query_pad_id=getattr(stack.query_tokenizer, "pad_token_id", model.pad_id),
                            max_docs=cfg.data.max_docs)

    with open(args.corpus, encoding="utf-8") as f:
        corpus = {json.loads(line)["doc_id"]: json.loads(line)["text"] for line in f if line.strip()}

    # PISCO's generate_from_text requires every row in a call to carry the same
    # number of documents.  Group by document count rather than padding with
    # duplicates: padding would feed PISCO evidence QuRO never sees, and the
    # resulting disagreement would be an artefact of this script.
    by_width: dict[int, list] = {}
    for i in range(len(dataset)):
        item = dataset[i]
        by_width.setdefault(len(item["retrieved_doc_ids"]), []).append(item)

    rows, agree = [], 0
    for width, group in sorted(by_width.items()):
        for start in range(0, len(group), 4):
            items = group[start : start + 4]
            batch = move_to_device(collator(items), device)
            quro_answers = model.generate_answer(batch, max_new_tokens=args.max_new_tokens)

            documents = [[corpus[d] for d in x["retrieved_doc_ids"]] for x in items]
            pisco_answers = stack.cocom.generate_from_text(
                [x["query"] for x in items], documents, max_new_tokens=args.max_new_tokens)

            for item, quro, pisco in zip(items, quro_answers, pisco_answers):
                golds = item["raw"].get("answers") or [item["raw"]["answer"]]
                pisco = pisco.strip()
                same = metrics.normalize_answer(quro) == metrics.normalize_answer(pisco)
                agree += int(same)
                rows.append({"id": item["id"], "query": item["query"], "golds": golds,
                             "n_docs": width,
                             "quro_pisco_direct": quro, "pisco_official": pisco,
                             "identical": same,
                             "quro_sub": metrics.score(quro, golds)["substring"],
                             "pisco_sub": metrics.score(pisco, golds)["substring"]})

    quro_sub = sum(r["quro_sub"] for r in rows) / len(rows)
    pisco_sub = sum(r["pisco_sub"] for r in rows) / len(rows)
    print(f"\nrows={len(rows)}  identical answers: {agree}/{len(rows)} ({agree/len(rows):.1%})")
    print(f"QuRO(pisco_direct) substring={quro_sub:.2%}   PISCO official substring={pisco_sub:.2%}")
    for row in [r for r in rows if not r["identical"]][:4]:
        print(f"\n  [differs, n_docs={row['n_docs']}] Q: {row['query'][:70]}"
              f"\n    quro : {row['quro_pisco_direct'][:100]}"
              f"\n    pisco: {row['pisco_official'][:100]}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"identical": agree, "n": len(rows), "quro_substring": quro_sub,
                       "pisco_substring": pisco_sub, "rows": rows}, f,
                      ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
