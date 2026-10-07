"""Offline, sentence-level visible evidence targets for QER training."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import doc_id_for, load_corpus, read_jsonl
from src.evidence import annotate_evidence, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--pisco_path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.resolve() == Path(args.train_file).resolve():
        raise ValueError("target preparation must not overwrite original rows")
    with open(Path(args.cache_dir)/"manifest.json") as handle:
        manifest = json.load(handle)
    with open(Path(args.pisco_path)/"config.json") as handle:
        cfg = json.load(handle)
    if (cfg.get("doc_max_length") != 128 or cfg.get("compr_rate") != 16
            or cfg.get("compr_model_name") is not None or cfg.get("kbtc_training")):
        raise ValueError("expected published PISCO 128/16 decoder-as-encoder checkpoint")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["decoder_model_name"], use_fast=True,
                                        padding_side="left", truncation_side="right")
    memories = [f"<MEM{i}>" for i in range(8)] if cfg.get("different_mem_tokens") else ["<MEM>"]
    tok.add_special_tokens({"additional_special_tokens": memories+["<AE>", "<ENC>", "<SEP>"]})
    corpus = load_corpus([args.corpus])
    for doc_id, text in corpus.items():
        if doc_id.startswith("d:") and doc_id_for(text) != doc_id:
            raise ValueError(f"{doc_id}: corpus/cache content hash mismatch")
    counts, retained = Counter(), {}
    cache_digest = digest(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in read_jsonl(args.train_file):
            # Strip query to match adapt_row in data.py, so source_digest stays stable.
            if "query" in row and row["query"] != row["query"].strip():
                row = {**row, "query": row["query"].strip()}
            annotated = annotate_evidence(row, corpus, tok, manifest, retained, cache_digest)
            a = annotated["evidence_annotation"]
            counts["rows"] += 1
            counts["rows_with_visible_target"] += bool(a["text"])
            counts["rows_all_facts_visible"] += a["all_facts_visible"]
            counts["support_facts"] += len(a["fact_visible"])
            counts["visible_facts"] += sum(a["fact_visible"])
            handle.write(json.dumps(annotated, ensure_ascii=False)+"\n")
    report = dict(counts, cache_digest=cache_digest, train_file=args.train_file,
                  output=str(output), tokenizer_source=cfg["decoder_model_name"])
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
