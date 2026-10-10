"""CPU-only preparation for matched open-domain RAG experiments.

Evaluation questions retain their original IDs/labels, while retrieval IDs are
namespaced by split. Native Hotpot contexts/support ranks never enter the new
retrieved evidence files. K5 is always the prefix of the SAME reranked K10 list.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re

import prepare_public_qa as public


def named_paths(values):
    result = {}
    for value in values:
        name, sep, path = value.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in result:
            raise ValueError(f"expected unique split_name=/path/to/file.jsonl: {value}")
        result[name] = Path(path)
    return result


def save_manifest(out, value):
    (out / "manifest.json").write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(value, ensure_ascii=False, indent=2))


def eval_queries(args):
    inputs = named_paths(args.inputs)
    out = public.ensure_output(args.out_dir)
    all_rows, counts = [], {}
    for name, path in inputs.items():
        rows, ids = [], set()
        for original in public.read_jsonl(path):
            identifier = str(original.get("id", original.get("q_id", ""))).strip()
            query = public.question(original)
            answers = public.strings(original.get("answers", original.get("answer", original.get("label"))))
            if not identifier or identifier in ids or not query or not answers:
                raise ValueError(f"{path}: missing/duplicate ID, query or gold answer: {identifier}")
            ids.add(identifier)
            retrieval_id = f"{name}::{identifier}"
            row = {"id": retrieval_id, "source_id": retrieval_id, "original_id": identifier,
                   "query": query, "answers": answers, "eval_split": name,
                   "source": original.get("source", name), "source_split": "evaluation"}
            if original.get("hop_type", original.get("type")):
                row["hop_type"] = original.get("hop_type", original.get("type"))
            rows.append(row)
        if not rows:
            raise ValueError(f"empty evaluation file: {path}")
        public.write_jsonl(out / f"{name}.queries.jsonl", rows)
        all_rows.extend(rows)
        counts[name] = len(rows)
    public.write_jsonl(out / "eval.queries.jsonl", all_rows)
    save_manifest(out, {"stage": "eval_questions_only", "counts": counts,
                        "inputs": public.file_records(inputs.values()),
                        "native_contexts_copied": False})


def attach_eval(args):
    root = Path(args.queries_dir)
    rows = list(public.read_jsonl(root / "eval.queries.jsonl"))
    index = public.RetrievalIndex(args.retrieval_jsonl)
    ranked, corpus = [], {}
    ks = sorted(set(int(k) for k in args.ks.split(",")))
    if not ks or min(ks) < 1:
        raise ValueError("--ks must contain positive document counts")
    # Resolve completely before creating output files; no partial benchmark.
    for row in rows:
        value, matched = index.get(row)
        if value is None:
            raise ValueError(f"missing retrieval: {row['id']}")
        ids = public.retrieve_documents(value, max(ks), corpus)
        if len(ids) != max(ks) or len(set(ids)) != len(ids):
            raise ValueError(f"need {max(ks)} distinct ranked documents: {row['id']}")
        ranked.append((row, ids, matched))
    required = {d for _, ids, _ in ranked for d in ids}
    if required - corpus.keys():
        raise ValueError("evaluation attach needs inlined document texts, not IDs alone")
    out = public.ensure_output(args.out_dir)
    for k in ks:
        directory = out / f"k{k}"
        directory.mkdir()
        splits = {}
        for row, ids, matched in ranked:
            # Only evidence IDs change; official gold aliases and original
            # question IDs are preserved for pairing with native-context runs.
            ready = {**row, "id": row["original_id"], "retrieved_doc_ids": ids[:k],
                     "evidence_protocol": f"kilt128_retrieved_k{k}", "retrieval_join": matched}
            splits.setdefault(row["eval_split"], []).append(ready)
        for name, values in splits.items():
            public.write_jsonl(directory / f"{name}.jsonl", values)
    public.write_jsonl(out / "corpus.jsonl", ({"doc_id": d, "text": corpus[d]} for d in sorted(required)))
    save_manifest(out, {"stage": "eval_retrieval_attached", "ks": ks,
                        "counts": dict(Counter(r["eval_split"] for r in rows)),
                        "required_documents": len(required), "k5_is_k10_prefix": 5 in ks and 10 in ks,
                        "query_inputs": public.file_records([root / "eval.queries.jsonl"]),
                        "retrieval_inputs": public.file_records(args.retrieval_jsonl)})


def missing_queries(args):
    root = Path(args.queries_dir)
    index = public.RetrievalIndex(args.retrieval_jsonl)
    out = public.ensure_output(args.out_dir)
    counts = {}
    for split in ("train", "dev"):
        pending = []
        for row in public.read_jsonl(root / f"{split}.queries.jsonl"):
            try:
                found, _ = index.get(row)
            except ValueError as error:
                # Several old rows share this normalised question: retrieve it
                # afresh so attach joins by ID instead of guessing among them.
                if "ambiguous" not in str(error):
                    raise
                found = None
                counts[f"{split}_ambiguous"] = counts.get(f"{split}_ambiguous", 0) + 1
            if found is None:
                pending.append(row)
                continue
            ids = public.retrieve_documents(found, args.max_docs, {})
            if len(ids) < args.max_docs:
                raise ValueError(f"existing retrieval has fewer than K{args.max_docs}: {row['id']}; use a new complete retrieval")
        public.write_jsonl(out / f"{split}.queries.jsonl", pending)
        counts[split] = len(pending)
    save_manifest(out, {"stage": "missing_queries_only", "counts": counts,
                        "query_inputs": public.file_records([root / f"{s}.queries.jsonl" for s in ("train", "dev")]),
                        "retrieval_inputs": public.file_records(args.retrieval_jsonl)})


def subset_ready(args):
    root = Path(args.full_ready)
    full = {r["id"]: r for r in public.read_jsonl(root / "train.jsonl")}
    questions = list(public.read_jsonl(Path(args.small_queries) / "train.queries.jsonl"))
    small = []
    for row in questions:
        found = full.get(row["id"])
        if found is None or public.question(found) != public.question(row) or found["answers"] != row["answers"]:
            raise ValueError(f"small cohort is not an identical labelled subset of full training: {row['id']}")
        small.append(found)
    if not small or len({r["id"] for r in small}) != len(small):
        raise ValueError("empty/duplicate small cohort")
    out = public.ensure_output(args.out_dir)
    public.write_jsonl(out / "train.jsonl", small)
    # Both cohorts use byte-for-byte identical dev questions/evidence/order.
    (out / "dev.jsonl").write_bytes((root / "dev.jsonl").read_bytes())
    save_manifest(out, {"stage": "nested_ready_subset", "training_ready": True,
                        "train_rows": len(small), "full_train_rows": len(full),
                        "train_by_source": dict(Counter(r["source"] for r in small)),
                        "inputs": public.file_records([root / "train.jsonl", root / "dev.jsonl",
                                                        Path(args.small_queries) / "train.queries.jsonl"]),
                        "files": public.file_records([out / "train.jsonl", out / "dev.jsonl"])})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("eval-queries")
    p.add_argument("--inputs", nargs="+", required=True, help="split_name=/path.jsonl")
    p.add_argument("--out_dir", required=True)
    p.set_defaults(func=eval_queries)
    p = sub.add_parser("attach-eval")
    p.add_argument("--queries_dir", required=True)
    p.add_argument("--retrieval_jsonl", action="append", required=True)
    p.add_argument("--ks", default="5,10")
    p.add_argument("--out_dir", required=True)
    p.set_defaults(func=attach_eval)
    p = sub.add_parser("missing")
    p.add_argument("--queries_dir", required=True)
    p.add_argument("--retrieval_jsonl", action="append", required=True)
    p.add_argument("--max_docs", type=int, default=5)
    p.add_argument("--out_dir", required=True)
    p.set_defaults(func=missing_queries)
    p = sub.add_parser("subset")
    p.add_argument("--full_ready", required=True)
    p.add_argument("--small_queries", required=True)
    p.add_argument("--out_dir", required=True)
    p.set_defaults(func=subset_ready)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
