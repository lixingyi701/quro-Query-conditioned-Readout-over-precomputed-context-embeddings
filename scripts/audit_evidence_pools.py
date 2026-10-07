"""Audit train-only pool/question/target diversity; no model or GPU required."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import statistics
from pathlib import Path


def pool_stats(groups):
    sizes = [len({r["query"].strip() for r in rows}) for rows in groups.values()]
    eligible = [rows for rows in groups.values()
                if len({r["query"].strip() for r in rows}) > 1]
    target = lambda r: r.get("evidence_annotation", {}).get("text") or "\n".join(
        f["text"].strip() for f in r.get("supporting_sentences", []))
    varying = [rows for rows in eligible if len({target(r) for r in rows if target(r)}) > 1]
    total = sum(len(rows) for rows in groups.values())
    return {"pools": len(groups), "distinct_questions_per_pool": dict(sorted(Counter(sizes).items())),
            "median_distinct_questions": statistics.median(sizes) if sizes else None,
            "multi_question_pools": len(eligible),
            "rows_in_multi_question_pools": sum(map(len, eligible)),
            "row_fraction_multi_question": sum(map(len, eligible))/max(1, total),
            "multi_question_pools_with_distinct_targets": len(varying),
            "rows_in_pools_with_distinct_targets": sum(map(len, varying))}


def audit(rows):
    ordered, unordered, documents = defaultdict(list), defaultdict(list), defaultdict(list)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate training IDs; resolve them before auditing")
    for row in rows:
        docs = row["retrieved_doc_ids"]
        if not docs:
            raise ValueError(f"{row['id']}: empty document pool")
        ordered[tuple(docs)].append(row)
        unordered[tuple(sorted(docs))].append(row)
        for doc_id in set(docs):
            documents[doc_id].append(row)
    return {"questions": len(rows), "ordered_pool": pool_stats(ordered),
            "rows_with_duplicate_document_ids": sum(len(set(r['retrieved_doc_ids'])) != len(r['retrieved_doc_ids']) for r in rows),
            "unordered_pool": pool_stats(unordered), "document_reuse": pool_stats(documents),
            "rows_with_support": sum(bool(r.get("supporting_sentences")) for r in rows),
            "rows_with_visible_target": sum(bool(r.get("evidence_annotation", {}).get("text")) for r in rows),
            "interpretation": "Exact-pool variation tests pool shortcuts; document reuse is a separate statistic. "
                              "Median 1 is a risk signal, not evidence that query supervision is impossible."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = Path(args.train_file)
    data = source.read_bytes()
    report = audit([json.loads(line) for line in data.decode().splitlines() if line.strip()])
    report.update(source_file=str(source.resolve()), source_sha256=hashlib.sha256(data).hexdigest())
    output = Path(args.output)
    if source.resolve() == output.resolve():
        raise ValueError("audit output must not overwrite training data")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
