"""Split any query file that embeds document text into a corpus plus ID-only queries.

Needed for evaluation sets such as ``trivia_qa_eval.jsonl``, which ship the
retrieved passages inline.  QuRO's online path only accepts document IDs, so the
text has to be lifted out, deduplicated and compressed once -- the same treatment
the training corpus gets.

Document IDs are content hashes (``src.data.doc_id_for``), so a passage shared
between two files lands in the cache exactly once and can be reused across them.

    python scripts/corpus_from_queries.py \
      --input /home/lxy/selecom/data/trivia_qa/trivia_qa_eval.jsonl \
      --out_dir /data02/quro/data/trivia --max_docs 10 --limit 2000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import _document_text, doc_id_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_docs", type=int, default=None, help="keep the top-k passages")
    ap.add_argument("--limit", type=int, default=None, help="keep only the first N queries")
    ap.add_argument("--prefix", default="q", help="ID prefix for generated query IDs")
    ap.add_argument("--distractors", type=int, default=0,
                    help="random passages added so that budget B < K*m and the "
                         "readout actually has to select; report this setting")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    corpus: dict[str, str] = {}
    queries = []

    with open(args.input, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.limit is not None and len(queries) >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            documents = row.get("documents") or ([row["document"]] if "document" in row else [])
            documents = [_document_text(d) for d in documents]
            if args.max_docs:
                documents = documents[: args.max_docs]
            documents = [d for d in documents if d.strip()]
            query = str(row.get("query", row.get("question", ""))).strip()
            answers = row.get("answers", row.get("answer"))
            answers = answers if isinstance(answers, list) else [answers]
            answers = [str(a) for a in answers if a is not None and str(a).strip()]
            if not (documents and query and answers):
                continue

            doc_ids = []
            for text in documents:
                key = doc_id_for(text)
                corpus.setdefault(key, text.strip())
                doc_ids.append(key)
            queries.append({"id": f"{args.prefix}-{i}", "query": query,
                            "retrieved_doc_ids": doc_ids, "answers": answers})

    # Padding to a fixed K makes the budget genuinely binding.  This is a noisy
    # top-k setting, not stock TriviaQA RAG, and must be reported as such.
    if args.distractors:
        rng = random.Random(args.seed)
        pool = list(corpus)
        for row in queries:
            gold = row["retrieved_doc_ids"]
            picked, seen = [], set(gold)
            while len(picked) < args.distractors and len(seen) < len(pool):
                candidate = rng.choice(pool)
                if candidate not in seen:
                    seen.add(candidate)
                    picked.append(candidate)
            position = rng.randrange(len(picked) + 1)
            row["retrieved_doc_ids"] = picked[:position] + gold + picked[position:]
            row["gold_rank"] = position
            row["n_distractors"] = len(picked)

    corpus_path = os.path.join(args.out_dir, "corpus.jsonl")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for doc_id, text in corpus.items():
            f.write(json.dumps({"doc_id": doc_id, "text": text}, ensure_ascii=False) + "\n")
    query_path = os.path.join(args.out_dir, "queries.jsonl")
    with open(query_path, "w", encoding="utf-8") as f:
        for row in queries:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mean_k = sum(len(r["retrieved_doc_ids"]) for r in queries) / max(1, len(queries))
    print(json.dumps({
        "queries": len(queries), "unique_documents": len(corpus),
        "mean_docs_per_query": round(mean_k, 2),
        "distractors": args.distractors,
        "estimated_cache_gb_at_m8_fp16": round(len(corpus) * 8 * 4096 * 2 / 1e9, 3),
        "corpus": corpus_path, "queries_file": query_path,
    }, indent=2))


if __name__ == "__main__":
    main()
