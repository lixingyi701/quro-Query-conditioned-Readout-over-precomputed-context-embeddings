"""Widen HotpotQA's retrieved set to K=20/50 with BM25-retrieved distractors.

Why this exists.  At K=10, P feeds 10*m = 80 soft tokens and beats QuRO by 11.7
EM.  But P's budget is K*m and grows with the retrieved set, while QuRO's is B
and does not -- so K=10 is the single worst case for the efficiency argument, and
the comparison as run says nothing about the regime the method is designed for.
Widening K is the only experiment that can move that number.

Two constraints shape the design:

**Distractors must come from the existing corpus.**  The latent cache covers the
260,236 paragraphs reachable from the 30k train / 7,405 validation questions.  A
distractor from outside it would need the cache rebuilt, so retrieval is
restricted to that set.  This is a real corpus, not a toy one.

**Distractors must be topical, not random.**  Random paragraphs are trivially
separable -- cosine top-B would dodge them and S would look far better than it
is.  BM25 against the question returns paragraphs that share its vocabulary,
which is the same kind of noise HotpotQA's own 8 distractors are (they were
TF-IDF retrieved).  warning_and_target §4.2 asks for topical distractors and
standard retrieval top-k, reported apart from the gold+random setting.

The gold paragraphs are always kept and re-inserted at random positions, so
``gold_ranks`` stays meaningful and position carries no signal.  A question whose
gold is unreachable is not silently dropped: the count is reported.

    python scripts/make_k_sweep_data.py --k 20 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(corpus_rows, max_features=400_000):
    """TF-IDF over the corpus, L2-normalised so a dot product is cosine."""
    texts = [row["text"] for row in corpus_rows]
    vectorizer = TfidfVectorizer(
        lowercase=True, sublinear_tf=True, max_features=max_features,
        stop_words="english", norm="l2", dtype=np.float32)
    matrix = vectorizer.fit_transform(texts)
    print(f"[index] {matrix.shape[0]} documents x {matrix.shape[1]} terms, "
          f"{matrix.nnz/1e6:.1f}M non-zeros")
    return vectorizer, matrix


def retrieve(vectorizer, matrix, queries, top_n, batch=256):
    """Top-n corpus rows per query, by cosine on TF-IDF."""
    out = np.empty((len(queries), top_n), dtype=np.int32)
    for start in range(0, len(queries), batch):
        chunk = queries[start : start + batch]
        scores = (vectorizer.transform(chunk) @ matrix.T).toarray()
        # argpartition then sort only the top slice: full sort over 260k columns
        # per query would dominate the runtime for no benefit.
        idx = np.argpartition(-scores, top_n, axis=1)[:, :top_n]
        order = np.argsort(-np.take_along_axis(scores, idx, axis=1), axis=1)
        out[start : start + len(chunk)] = np.take_along_axis(idx, order, axis=1)
        if start % (batch * 20) == 0:
            print(f"  retrieved {start + len(chunk)}/{len(queries)}", flush=True)
    return out


def widen(rows, retrieved, doc_ids, k, seed):
    """Keep gold, fill to K with the highest-ranked non-gold, shuffle positions."""
    rng = random.Random(seed)
    out, short = [], 0
    for row, ranked in zip(rows, retrieved):
        gold = [row["retrieved_doc_ids"][i] for i in row["gold_ranks"]]
        gold_set = set(gold)
        # The question's original 8 distractors are kept ahead of the BM25 ones:
        # they are the dataset's own hard negatives and dropping them would make
        # the wider setting easier, not harder, which is the opposite of the point.
        pool, seen = [], set(gold_set)
        for doc_id in row["retrieved_doc_ids"]:
            if doc_id not in seen:
                pool.append(doc_id); seen.add(doc_id)
        for index in ranked:
            if len(gold) + len(pool) >= k:
                break
            doc_id = doc_ids[index]
            if doc_id not in seen:
                pool.append(doc_id); seen.add(doc_id)

        picked = gold + pool[: max(0, k - len(gold))]
        if len(picked) < k:
            short += 1
        rng.shuffle(picked)
        new = dict(row)
        new["retrieved_doc_ids"] = picked
        new["gold_ranks"] = sorted(picked.index(g) for g in gold)
        new["n_gold"] = len(gold)
        new["n_distractors"] = len(picked) - len(gold)
        new["k"] = len(picked)
        new["distractor_source"] = "hotpot_native+bm25"
        out.append(new)
    return out, short


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data02/quro/data/hotpot")
    ap.add_argument("--k", type=int, nargs="+", default=[20, 50])
    ap.add_argument("--splits", nargs="+", default=["train", "dev"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    corpus = read_jsonl(os.path.join(args.data, "corpus.jsonl"))
    doc_ids = [row["doc_id"] for row in corpus]
    vectorizer, matrix = build_index(corpus)

    largest = max(args.k)
    # Retrieve a margin above the largest K: some hits are the question's own
    # gold or existing distractors and get filtered out.
    top_n = largest + 24

    summary = {"corpus_documents": len(corpus), "splits": {}}
    for split in args.splits:
        rows = read_jsonl(os.path.join(args.data, f"{split}.jsonl"))
        print(f"[{split}] {len(rows)} questions")
        ranked = retrieve(vectorizer, matrix, [r["query"] for r in rows], top_n)
        for k in args.k:
            widened, short = widen(rows, ranked, doc_ids, k, args.seed)
            path = os.path.join(args.data, f"{split}_k{k}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for row in widened:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            sizes = {}
            for row in widened:
                sizes[row["k"]] = sizes.get(row["k"], 0) + 1
            info = {"questions": len(widened), "short_of_k": short, "k_sizes": sizes}
            summary["splits"][f"{split}_k{k}"] = info
            print(f"  wrote {path}: {info}")

    with open(os.path.join(args.data, "k_sweep_stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
