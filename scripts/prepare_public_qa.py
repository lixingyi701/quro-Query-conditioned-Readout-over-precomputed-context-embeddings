"""Reuse the published COCOM/PISCO question pool, then attach real retrieval.

``export`` downloads the pinned 56.6 MB public parquet or reads a local copy.
``attach`` joins existing retrieved passages/IDs without changing their order.
Question-only files are deliberately named *.queries.jsonl: they cannot yet be
fed to QuRODataset. No random distractors, gold-context substitution, teacher
generation, or implicit replacement of gold targets happens in this script.

Only HF download/parquet input needs requirements-data.txt; JSONL workflows use
the standard library and do not import torch or load a model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import unicodedata


PUBLIC_REPO = "dmrau/multi_qa"
PUBLIC_REVISION = "b0b01e2a0f6e251e9cdd191f5018bec7b170a554"
PUBLIC_PARQUET = "data/train-00000-of-00001.parquet"
SOURCES = ("nq_open", "triviaqa", "hotpotqa", "asqa", "msmarco",
           "adversarial_qa", "wikiqa", "wiki_qa", "sciq", "freebase_qa", "squad")


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_no}: expected an object")
                yield value


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def digest_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_records(paths):
    return [{"path": str(Path(p).resolve()), "sha256": digest_file(p)} for p in paths]


def normalized_question(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = "".join(" " if unicodedata.category(c).startswith("P") else c for c in value)
    return " ".join(value.split())


def question(row):
    value = row.get("query", row.get("question", row.get("content", "")))
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("question must be a string")
    return value.strip()


def row_id(row):
    value = row.get("source_id", row.get("id", row.get("q_id", "")))
    return "" if value is None else str(value).strip()


def strings(value):
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple)) else [value]
    if any(not isinstance(v, str) for v in values):
        raise ValueError("expected string answer labels")
    return [v for v in values if v.strip()]


def source_for(identifier):
    for source in sorted(SOURCES, key=len, reverse=True):
        if re.fullmatch(re.escape(source) + r"\d+", identifier):
            return "wikiqa" if source == "wiki_qa" else source
    raise ValueError(f"unknown public source ID {identifier!r}; supply a supported public pool")


def rank_key(seed, namespace, value):
    return hashlib.sha256(f"{seed}\n{namespace}\n{value}".encode()).hexdigest()


def load_public(args):
    if args.input_jsonl:
        return list(read_jsonl(args.input_jsonl)), {"kind": "local_jsonl", "files": file_records([args.input_jsonl])}
    path = args.input_parquet
    provenance = {"kind": "local_parquet"}
    if not path:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as error:
            raise RuntimeError("Install python -m pip install -r requirements-data.txt") from error
        path = hf_hub_download(repo_id=PUBLIC_REPO, filename=PUBLIC_PARQUET,
                               repo_type="dataset", revision=args.revision,
                               cache_dir=args.hf_cache_dir, local_files_only=args.local_only)
        # A branch/tag is resolved to an immutable snapshot by hf_hub_download.
        revision = Path(path).parent.parent.name
        provenance = {"kind": "huggingface", "repo": PUBLIC_REPO,
                      "requested_revision": args.revision, "resolved_revision": revision}
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet input requires python -m pip install -r requirements-data.txt") from error
    parquet = pq.ParquetFile(path)
    rows = [r for batch in parquet.iter_batches(batch_size=4096) for r in batch.to_pylist()]
    provenance["files"] = file_records([path])
    return rows, provenance


def export_rows(raw, *, exclude_paths=(), sources=(), seed=42, dev_fraction=.01, limit=None):
    excluded_ids, excluded_questions = set(), set()
    for path in exclude_paths:
        for row in read_jsonl(path):
            if row_id(row):
                excluded_ids.add(row_id(row))
            if question(row):
                excluded_questions.add(normalized_question(question(row)))
    groups = defaultdict(list)
    stats = Counter()
    seen_ids = set()
    for value in raw:
        stats["input_rows"] += 1
        identifier, query = row_id(value), question(value)
        if not identifier:
            raise ValueError("public row has no source ID")
        if identifier in seen_ids:
            raise ValueError(f"duplicate public ID {identifier}")
        seen_ids.add(identifier)
        source = source_for(identifier)
        answers = strings(value.get("label", value.get("answers", value.get("answer"))))
        if not query or not answers or not normalized_question(query):
            stats["invalid_rows"] += 1
            continue
        if sources and source not in sources:
            stats["source_filtered_rows"] += 1
            continue
        norm = normalized_question(query)
        if identifier in excluded_ids or norm in excluded_questions:
            stats["evaluation_overlap_removed"] += 1
            continue
        # Keep every label/order; use the first label for our existing gold CE recipe.
        row = {"id": identifier, "source_id": identifier, "query": query,
               "answers": answers, "source": source, "source_split": "train",
               "training_target_policy": "first_public_label"}
        groups[norm].append(row)
    train, dev = [], []
    threshold = int(dev_fraction * (1 << 256))
    for norm, rows in groups.items():
        partition = dev if int(rank_key(seed, "split", norm), 16) < threshold else train
        partition.extend(rows)
    # Sample only training, after assigning entire question groups to splits.
    # Uniform sampling retains the published mixture in expectation; no quotas.
    train.sort(key=lambda r: (rank_key(seed, "sample", r["id"]), r["id"]))
    stats["available_train_rows"] = len(train)
    if limit is not None:
        train = train[:limit]
    train.sort(key=lambda r: (rank_key(seed, "order", r["id"]), r["id"]))
    dev.sort(key=lambda r: (rank_key(seed, "order", r["id"]), r["id"]))
    stats.update({"train_rows": len(train), "dev_rows": len(dev)})
    return train, dev, {"counts": dict(stats), "train_by_source": dict(Counter(r["source"] for r in train)),
                        "dev_by_source": dict(Counter(r["source"] for r in dev)),
                        "evaluation_exclusion_checked": bool(exclude_paths)}


def ensure_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"output directory must be empty: {path}; choose a new directory")
    return path


def export_command(args):
    out = ensure_output(args.out_dir)
    raw, provenance = load_public(args)
    selected = tuple(x.strip() for x in args.sources.split(",") if x.strip())
    if set(selected) - set(SOURCES):
        raise ValueError(f"unknown --sources: {set(selected) - set(SOURCES)}")
    selected = tuple("wikiqa" if s == "wiki_qa" else s for s in selected)
    train, dev, stats = export_rows(raw, exclude_paths=args.exclude_jsonl, sources=selected,
                                   seed=args.seed, dev_fraction=args.dev_fraction, limit=args.limit)
    if not train:
        raise ValueError("no training rows left")
    write_jsonl(out / "train.queries.jsonl", train)
    write_jsonl(out / "dev.queries.jsonl", dev)
    stats.update({"stage": "questions_only", "input": provenance, "seed": args.seed,
                  "dev_fraction": args.dev_fraction, "limit": args.limit, "sources": selected,
                  "exclusions": file_records(args.exclude_jsonl), "target": "first_public_label",
                  "retrieval_attached": False,
                  "files": file_records([out / "train.queries.jsonl", out / "dev.queries.jsonl"])})
    (out / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


def document_text(value):
    # Same title/text formatting and content ID as src.data, without importing torch.
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        title = str(value.get("title", "")).strip()
        text = str(value.get("text", value.get("content", value.get("document", ""))))
        return (f"Title: {title}\nContent: {text}" if title else text).strip()
    raise ValueError("retrieved document must be text or an object")


def content_id(text):
    return "d:" + hashlib.sha1(text.strip().encode()).hexdigest()[:20]


class RetrievalIndex:
    def __init__(self, paths):
        self.by_id, self.by_question = {}, defaultdict(list)
        for path in paths:
            for row in read_jsonl(path):
                identifier, query = row_id(row), question(row)
                if not identifier and not query:
                    raise ValueError(f"{path}: retrieval row has no ID or question")
                if identifier:
                    if identifier in self.by_id and self.by_id[identifier] != row:
                        raise ValueError(f"conflicting retrieval for {identifier}")
                    self.by_id[identifier] = row
                if query:
                    norm = normalized_question(query)
                    if row not in self.by_question[norm]:
                        self.by_question[norm].append(row)

    def get(self, row):
        value = self.by_id.get(row["source_id"])
        norm = normalized_question(row["query"])
        if value is not None:
            if question(value) and normalized_question(question(value)) != norm:
                raise ValueError(f"retrieval ID matches but question differs: {row['id']}")
            return value, "source_id"
        values = self.by_question.get(norm, [])
        if len(values) > 1:
            # Duplicate questions across sources are harmless when every
            # candidate carries the same ordered evidence.
            rankings = {json.dumps(retrieve_documents(v, 10 ** 6, {}), ensure_ascii=False) for v in values}
            if len(rankings) == 1 and retrieve_documents(values[0], 10 ** 6, {}):
                return values[0], "normalized_question"
            raise ValueError(f"ambiguous question-based retrieval join: {row['id']}")
        return (values[0], "normalized_question") if values else (None, None)


def retrieve_documents(value, max_docs, corpus):
    ids = value.get("retrieved_doc_ids", value.get("doc_ids"))
    docs = value.get("documents", value.get("docs", value.get("contexts")))
    if docs is None and "document" in value:
        docs = [value["document"]]
    if docs is not None and not isinstance(docs, list):
        raise ValueError("retrieved documents must be an ordered list")
    if ids is not None and (not isinstance(ids, list) or any(x is None for x in ids)):
        raise ValueError("retrieved_doc_ids must be a list of non-null IDs")
    ids = [str(x) for x in ids] if ids is not None else None
    if ids is not None and docs is not None and len(ids) != len(docs):
        raise ValueError("document text/ID counts disagree")
    result = []
    for rank in range(min(max_docs, len(ids if ids is not None else (docs or [])))):
        doc = docs[rank] if docs is not None else None
        text = document_text(doc) if doc is not None else None
        if text is not None and not text:
            raise ValueError("empty retrieved passage")
        explicit = next((doc[k] for k in ("doc_id", "id", "passage_id", "_id") if doc.get(k) is not None), None) if isinstance(doc, dict) else None
        identifier = ids[rank] if ids is not None else (str(explicit) if explicit is not None else content_id(text))
        if not identifier.strip():
            raise ValueError("empty retrieved document ID")
        if text is not None:
            if identifier in corpus and corpus[identifier] != text:
                raise ValueError(f"conflicting text for document {identifier}")
            corpus[identifier] = text
        # Deduplicate storage, not ranked references: an existing retriever may
        # return repeated text at different ranks. Preserve its actual input.
        result.append(identifier)
    return result


def attach_command(args):
    out = ensure_output(args.out_dir)
    root = Path(args.queries_dir)
    input_paths = [root / f"{split}.queries.jsonl" for split in ("train", "dev")]
    splits = {split: list(read_jsonl(path)) for split, path in zip(("train", "dev"), input_paths)}
    index, corpus = RetrievalIndex(args.retrieval_jsonl), {}
    missing, ready, joins = [], {"train": [], "dev": []}, Counter()
    for split, rows in splits.items():
        for row in rows:
            value, matched_by = index.get(row)
            if value is None:
                missing.append({**row, "split": split, "missing_reason": "no_retrieval_row"})
                continue
            ids = retrieve_documents(value, args.max_docs, corpus)
            if len(ids) < args.max_docs:
                missing.append({**row, "split": split, "missing_reason": "fewer_than_requested_docs", "found_docs": len(ids)})
                continue
            # Existing files may contain teacher answers or different labels. The
            # public gold labels remain authoritative for our projector CE recipe.
            ready[split].append({**row, "retrieved_doc_ids": ids,
                                 "retrieval_join": matched_by,
                                 "retrieval_source_id": row_id(value)})
            joins[matched_by] += 1
    required = {d for rows in ready.values() for row in rows for d in row["retrieved_doc_ids"]}
    cached = set()
    if args.cache_manifest:
        cached = set(json.loads(Path(args.cache_manifest).read_text())["documents"])
    for path in args.corpus_jsonl:
        for doc in read_jsonl(path):
            identifier = str(doc["doc_id"])
            if identifier in required:
                text = document_text(doc)
                if not text or (identifier in corpus and corpus[identifier] != text):
                    raise ValueError(f"empty/conflicting corpus text for {identifier}")
                corpus[identifier] = text
    missing_docs = sorted(required - set(corpus) - cached)
    complete = not missing and not missing_docs
    write_jsonl(out / "missing_retrieval.jsonl", missing)
    write_jsonl(out / "missing_documents.jsonl", ({"doc_id": d} for d in missing_docs))
    write_jsonl(out / "corpus.jsonl", ({"doc_id": d, "text": corpus[d]} for d in sorted(required & set(corpus))))
    # Never create train.jsonl from a partially joined pool. Its composition
    # would otherwise silently collapse to the already downloaded datasets.
    if complete:
        for split in ready:
            write_jsonl(out / f"{split}.jsonl", ready[split])
    stats = {"stage": "retrieval_attached" if complete else "retrieval_incomplete",
             "training_ready": complete, "max_docs": args.max_docs,
             "query_inputs": file_records(input_paths), "retrieval_inputs": file_records(args.retrieval_jsonl),
             "corpus_inputs": file_records(args.corpus_jsonl),
             "cache_manifest": file_records([args.cache_manifest]) if args.cache_manifest else [],
             "joins": dict(joins), "missing_queries": len(missing),
             "missing_queries_by_source": dict(Counter(r["source"] for r in missing)),
             "missing_documents": len(missing_docs), "required_documents": len(required),
             "documents_already_cached": len(required & cached),
             "documents_to_compress": len(required - cached),
             "train_rows": len(ready["train"]), "dev_rows": len(ready["dev"]),
             "train_by_source": dict(Counter(r["source"] for r in ready["train"])),
             "target": "first_public_label", "teacher_outputs_imported": False}
    stats["files"] = file_records(sorted(out.glob("*.jsonl")))
    (out / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if not complete:
        raise SystemExit("Retrieval incomplete; use missing_retrieval.jsonl/missing_documents.jsonl to finish preparation.")


def length_stats(lengths, cap):
    ordered = sorted(lengths)
    return {"rows": len(ordered), "max": max(ordered, default=0),
            "p50": ordered[int(.5 * (len(ordered) - 1))] if ordered else 0,
            "p95": ordered[int(.95 * (len(ordered) - 1))] if ordered else 0,
            "over_cap": sum(n > cap for n in ordered), "cap": cap}


def audit_command(args):
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Token-length audit requires the training environment's transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    report = {}
    for split in ("train", "dev"):
        lengths = defaultdict(lambda: {"query": [], "answer": []})
        for row in read_jsonl(Path(args.queries_dir) / f"{split}.queries.jsonl"):
            source = row["source"]
            # Exactly the text used by QuRODataset before slicing/EOS addition.
            lengths[source]["query"].append(len(tokenizer(row["query"], add_special_tokens=False)["input_ids"]))
            lengths[source]["answer"].append(len(tokenizer(" " + row["answers"][0].strip(), add_special_tokens=False)["input_ids"]))
        report[split] = {s: {"query": length_stats(v["query"], args.max_query_len),
                              "answer": length_stats(v["answer"], args.max_answer_len)} for s, v in lengths.items()}
    print(json.dumps(report, indent=2))
    if any(v[k]["over_cap"] for split in report.values() for v in split.values() for k in v):
        raise SystemExit("Lengths exceed the proposed caps; raise training caps before using this mixture.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="reuse the public question/label mixture")
    inputs = export.add_mutually_exclusive_group()
    inputs.add_argument("--input_jsonl", help="local id/content/label JSONL")
    inputs.add_argument("--input_parquet", help="local public parquet")
    export.add_argument("--revision", default=PUBLIC_REVISION)
    export.add_argument("--hf_cache_dir", default=os.path.join(os.environ.get("QURO_ROOT", "/data02/quro"), "hf-cache"))
    export.add_argument("--local_only", action="store_true")
    export.add_argument("--out_dir", required=True)
    export.add_argument("--exclude_jsonl", action="append", default=[], help="repeat for every tuning/evaluation question file")
    export.add_argument("--sources", default="", help="optional comma-separated source filter; default keeps the full mixture")
    export.add_argument("--limit", type=int, help="sample this many training rows; default all eligible rows")
    export.add_argument("--seed", type=int, default=42, help="data split/sample seed, independent of model seed")
    export.add_argument("--dev_fraction", type=float, default=.01)
    export.set_defaults(func=export_command)
    attach = sub.add_parser("attach", help="join fixed, ordered retrieval and build cache-first training files")
    attach.add_argument("--queries_dir", required=True)
    attach.add_argument("--retrieval_jsonl", action="append", required=True, help="repeat for existing per-dataset retrieval files")
    attach.add_argument("--corpus_jsonl", action="append", default=[], help="optional corpus for ID-only retrieval")
    attach.add_argument("--cache_manifest", help="existing cache manifest covers ID-only passages without source text")
    attach.add_argument("--out_dir", required=True)
    attach.add_argument("--max_docs", type=int, default=5)
    attach.set_defaults(func=attach_command)
    audit = sub.add_parser("audit", help="check real reader-tokenizer lengths before training")
    audit.add_argument("--queries_dir", required=True)
    audit.add_argument("--tokenizer_path", required=True, help="local frozen reader tokenizer")
    audit.add_argument("--max_answer_len", type=int, default=128)
    audit.add_argument("--max_query_len", type=int, default=256)
    audit.set_defaults(func=audit_command)
    args = parser.parse_args()
    if args.command == "export":
        if not 0 <= args.dev_fraction < 1 or (args.limit is not None and args.limit < 1):
            parser.error("require 0 <= dev_fraction < 1 and a positive --limit")
    elif args.command == "attach" and args.max_docs < 1:
        parser.error("--max_docs must be positive")
    elif args.command == "audit" and min(args.max_answer_len, args.max_query_len) < 1:
        parser.error("token caps must be positive")
    args.func(args)


if __name__ == "__main__":
    main()
