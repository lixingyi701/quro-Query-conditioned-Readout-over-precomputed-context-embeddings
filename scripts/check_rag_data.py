"""Validate cache-first QuRO JSONL before allocating a GPU.

Checks the online contract (query, retrieved IDs, targets, discrete budgets) and,
when supplied, verifies full coverage against a latent-cache manifest. This
script intentionally has no torch dependency.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def retrieved_ids(row, max_docs):
    explicit = row.get("retrieved_doc_ids", row.get("doc_ids"))
    if explicit is not None:
        return [str(value) for value in explicit][:max_docs]
    docs = row.get("documents")
    if docs is None and "document" in row:
        docs = [row["document"]]
    result = []
    row_id = str(row.get("id", row.get("q_id", "unknown")))
    for rank, doc in enumerate(list(docs or [])[:max_docs]):
        if isinstance(doc, dict):
            value = next((doc[key] for key in ("doc_id", "id", "passage_id", "_id")
                          if doc.get(key) is not None), None)
            result.append(str(value) if value is not None else f"{row_id}:doc:{rank}")
        else:
            result.append(f"{row_id}:doc:{rank}")
    return result


def load_cache_ids(cache_dir):
    if not cache_dir:
        return None, None
    path = os.path.join(cache_dir, "manifest.json")
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    return set(manifest.get("documents", {})), manifest


def check(path, cache_ids, buckets, max_docs):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path}: empty file")
    problems, missing_cache = [], Counter()
    teacher_count = budget_count = cache_complete_rows = 0
    k_values = []
    for index, row in enumerate(rows):
        row_id = str(row.get("id", row.get("q_id", index)))
        query = row.get("query", row.get("question"))
        ids = retrieved_ids(row, max_docs)
        answers = row.get("answers", row.get("answer", []))
        answers = answers if isinstance(answers, (list, tuple)) else [answers]
        teacher = row.get("teacher_output", row.get("teacher_answer"))
        if not isinstance(query, str) or not query.strip():
            problems.append(f"{row_id}: missing query")
        if not ids:
            problems.append(f"{row_id}: no retrieved document IDs")
        if not teacher and not any(str(value).strip() for value in answers):
            problems.append(f"{row_id}: no teacher_output or answer")
        if teacher:
            teacher_count += 1
        if row.get("budget") is not None:
            budget_count += 1
            try:
                budget = int(row["budget"])
            except (TypeError, ValueError):
                problems.append(f"{row_id}: budget is not an integer")
            else:
                if budget not in buckets:
                    problems.append(f"{row_id}: budget {budget} not in {sorted(buckets)}")
        if cache_ids is not None:
            row_missing = False
            for doc_id in ids:
                if doc_id not in cache_ids:
                    missing_cache[doc_id] += 1
                    row_missing = True
            cache_complete_rows += int(not row_missing)
        k_values.append(len(ids))

    print(f"\n=== {path} ===")
    print(f"rows={len(rows)} teacher={teacher_count/len(rows):.1%} "
          f"budget_labels={budget_count/len(rows):.1%} "
          f"K_median={sorted(k_values)[len(k_values)//2]} K_max={max(k_values)}")
    if cache_ids is not None:
        print(f"cache_coverage={cache_complete_rows/len(rows):.1%} "
              f"missing_unique={len(missing_cache)}")
    for problem in problems[:10]:
        print("ERROR", problem)
    for doc_id, count in missing_cache.most_common(10):
        print(f"ERROR cache missing {doc_id!r} (referenced {count}x)")
    if problems or missing_cache:
        raise SystemExit(f"validation failed: {len(problems)} row errors, "
                         f"{len(missing_cache)} missing cache IDs")
    print("PASS online data/cache contract")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--cache_dir")
    parser.add_argument("--max_docs", type=int, default=5)
    parser.add_argument("--budget_buckets", default="4,8,16,32")
    args = parser.parse_args()
    if args.max_docs < 1:
        raise ValueError("max_docs must be positive")
    buckets = {int(value) for value in args.budget_buckets.split(",") if value.strip()}
    cache_ids, manifest = load_cache_ids(args.cache_dir)
    if manifest:
        print(f"cache={manifest.get('compressor')} documents={len(cache_ids)} "
              f"shape=({manifest.get('latent_size')},{manifest.get('hidden_size')})")
    for path in args.files:
        check(path, cache_ids, buckets, args.max_docs)


if __name__ == "__main__":
    main()
