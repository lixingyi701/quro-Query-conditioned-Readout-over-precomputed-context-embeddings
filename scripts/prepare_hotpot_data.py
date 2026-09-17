"""Convert HotpotQA's distractor split into QuRO's cache-first corpus + query files.

Why HotpotQA, and why the distractor setting specifically.

TriviaQA is the worst case this method could have been evaluated on.  Its
questions are single-fact, so one well-chosen latent answers them and the other
B-1 output slots have nothing to do -- a multi-slot readout has no structural
reason to beat cosine top-B there, and measured it does not (S 71.70 vs C 70.90
at B=8).  HotpotQA's questions need two paragraphs combined, so the budget has
something to allocate and "which evidence, for this question" stops being a
one-slot decision.  That is the condition under which query-conditioned readout
should matter, if it matters at all.

The distractor setting hands each question its own 10 paragraphs -- 2 gold and 8
distractors, in shuffled order -- so no retriever sits between the dataset and
the measurement.  Retrieval noise is a separate variable and mixing it in here
would make a null result unattributable.

Output files follow the same contract as the TriviaQA/gonogo data:

* ``corpus.jsonl``          ``{"doc_id", "text"}``, globally deduplicated by content
  hash, so a paragraph shared by several questions is compressed and cached once;
* ``train/dev/test.jsonl``  ``{"id", "query", "retrieved_doc_ids", "answers", ...}``,
  the format ``src/data.py:adapt_row`` already reads.

``dev`` and ``test`` are disjoint halves of the official validation split.  The
TriviaQA 2000 has been looked at repeatedly and is development data by now
(HANDOFF.md §2); starting HotpotQA with the split already made is
cheaper than retrofitting one after the fact.

Paragraph order is left exactly as the dataset shuffled it.  ``gold_ranks``
records where the two gold paragraphs landed, so attention targeting can be
scored without assuming they are adjacent -- they generally are not.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import doc_id_for

# Identical to the existing corpus files, so a cache built from either is keyed
# the same way and the two datasets can share one compressor run.
DOC_TEMPLATE = "Title: {title}\nContent: {content}"


def paragraph_text(title: str, sentences) -> str:
    # HotpotQA sentences already carry their leading space, so concatenating is
    # correct; " ".join would double every inter-sentence space and change the
    # content hash against any cache built elsewhere.
    return DOC_TEMPLATE.format(title=str(title), content="".join(str(s) for s in sentences))


def convert_rows(frame: pd.DataFrame, prefix: str, corpus: dict) -> list:
    rows = []
    for _, item in frame.iterrows():
        context = item["context"]
        titles, sentences = list(context["title"]), list(context["sentences"])
        gold_titles = {str(t) for t in item["supporting_facts"]["title"]}

        doc_ids, gold_ranks = [], []
        for position, (title, sents) in enumerate(zip(titles, sentences)):
            text = paragraph_text(title, sents)
            doc_id = doc_id_for(text)
            corpus.setdefault(doc_id, text)
            doc_ids.append(doc_id)
            if str(title) in gold_titles:
                gold_ranks.append(position)

        answer = str(item["answer"])
        rows.append({
            "id": f"{prefix}-{item['id']}",
            "query": str(item["question"]),
            "retrieved_doc_ids": doc_ids,
            # HotpotQA ships one gold string per question, with no alias list.  EM
            # is therefore strictly harsher here than on TriviaQA, where any of
            # several aliases counts; the two numbers are not comparable.
            "answers": [answer],
            "gold_ranks": gold_ranks,
            "n_gold": len(gold_ranks),
            "n_distractors": len(doc_ids) - len(gold_ranks),
            "hop_type": str(item["type"]),          # bridge | comparison
            "level": str(item["level"]),
            # Comparison questions are largely yes/no.  A model can reach a fifth
            # of them by always saying "yes", so these have to be scorable apart
            # from the span questions rather than averaged in silently.
            "is_yes_no": answer.strip().lower() in {"yes", "no"},
        })
    return rows


def write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def describe(name: str, rows: list) -> dict:
    golds = Counter(r["n_gold"] for r in rows)
    return {
        "split": name,
        "questions": len(rows),
        "yes_no": sum(r["is_yes_no"] for r in rows),
        "hop_type": dict(Counter(r["hop_type"] for r in rows)),
        "docs_per_question": dict(Counter(len(r["retrieved_doc_ids"]) for r in rows)),
        "gold_per_question": dict(golds),
        # Adjacent golds would let the older contiguous-block diagnostics work
        # unchanged; report the share rather than assuming it.
        "adjacent_golds": sum(
            1 for r in rows if len(r["gold_ranks"]) == 2
            and abs(r["gold_ranks"][0] - r["gold_ranks"][1]) == 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/data02/quro/data/hotpot_raw/distractor")
    ap.add_argument("--out", default="/data02/quro/data/hotpot")
    ap.add_argument("--train_questions", type=int, default=30000,
                    help="0 for all 90447; the corpus and therefore the latent "
                         "cache grow roughly 7 unique paragraphs per question")
    ap.add_argument("--dev_questions", type=int, default=2000,
                    help="taken from the official validation split; the rest "
                         "becomes the held-out test split")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    random.seed(args.seed)

    train_frames = [pd.read_parquet(os.path.join(args.raw, name))
                    for name in sorted(os.listdir(args.raw)) if name.startswith("train")]
    train_all = pd.concat(train_frames, ignore_index=True)
    validation = pd.read_parquet(
        os.path.join(args.raw, [n for n in os.listdir(args.raw)
                                if n.startswith("validation")][0]))
    print(f"[raw] train={len(train_all)} validation={len(validation)}")

    if args.train_questions and args.train_questions < len(train_all):
        train_all = train_all.sample(n=args.train_questions, random_state=args.seed)
    # Shuffle before splitting: the validation file is not ordered randomly with
    # respect to hop type, so slicing it raw would skew dev against test.
    validation = validation.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    dev_n = min(args.dev_questions or len(validation), len(validation))

    corpus = {}
    splits = {
        "train": convert_rows(train_all, "hpqa", corpus),
        "dev": convert_rows(validation.iloc[:dev_n], "hpqa-dev", corpus),
        "test": convert_rows(validation.iloc[dev_n:], "hpqa-test", corpus),
    }

    stats = {"corpus_documents": len(corpus), "splits": []}
    for name, rows in splits.items():
        write_jsonl(os.path.join(args.out, f"{name}.jsonl"), rows)
        info = describe(name, rows)
        stats["splits"].append(info)
        print(f"[{name}] {info}")

    write_jsonl(os.path.join(args.out, "corpus.jsonl"),
                [{"doc_id": k, "text": v} for k, v in corpus.items()])
    # 8 KB per document at m=8, fp16, 4096-d -- the number that decides whether the
    # cache build is an hour or an afternoon, so print it before anyone starts one.
    stats["cache_gb_at_m8_fp16"] = round(len(corpus) * 8 * 4096 * 2 / 1e9, 1)
    with open(os.path.join(args.out, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[corpus] {len(corpus)} unique paragraphs "
          f"-> ~{stats['cache_gb_at_m8_fp16']} GB of latents at m=8, fp16")


if __name__ == "__main__":
    main()
