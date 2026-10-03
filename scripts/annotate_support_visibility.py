"""Offline support visibility annotation; raw documents never enter QA training."""
import argparse
from collections import Counter
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import parse_eval_files
from src.data import doc_id_for, load_corpus, read_jsonl
from src.support import annotate_row, manifest_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_files", required=True, help="train=a.jsonl,dev=b.jsonl,test=c.jsonl")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--pisco_path", required=True, help="same local published checkpoint as the cache")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    with open(os.path.join(args.cache_dir, "manifest.json")) as handle:
        manifest = json.load(handle)
    if not str(manifest.get("compressor", "")).lower().startswith("pisco"):
        raise ValueError("annotation supports the published decoder-as-encoder PISCO cache only")
    with open(os.path.join(args.pisco_path, "config.json")) as handle:
        config = json.load(handle)
    if (config.get("doc_max_length") != 128 or config.get("compr_rate") != 16
            or config.get("compr_model_name") is not None or config.get("kbtc_training")):
        raise ValueError("expected published PISCO 128/16 without a separate/query encoder")
    # Mirror COCOM.create_decoder_tokenizer without loading any model weights.
    tokenizer = AutoTokenizer.from_pretrained(config["decoder_model_name"], use_fast=True,
                                              padding_side="left")
    tokens = ([f"<MEM{i}>" for i in range(8)] if config.get("different_mem_tokens") else ["<MEM>"])
    tokenizer.add_special_tokens({"additional_special_tokens": tokens+["<AE>", "<ENC>", "<SEP>"]})
    corpus = load_corpus([args.corpus])
    digest, retained_ends = manifest_digest(manifest), {}
    for doc_id, text in corpus.items():
        if doc_id.startswith("d:") and doc_id_for(text) != doc_id:
            raise ValueError(f"{doc_id}: corpus text does not match its content-addressed cache ID")
    os.makedirs(args.out_dir, exist_ok=True)
    report = {"pisco_path": args.pisco_path, "cache_dir": args.cache_dir, "splits": {}}
    for name, path in parse_eval_files(args.input_files).items():
        output_path = os.path.join(args.out_dir, name+".jsonl")
        if os.path.abspath(output_path) == os.path.abspath(path):
            raise ValueError("annotation output must not overwrite its input")
        stats = Counter()
        with open(output_path, "w", encoding="utf-8") as handle:
            for row in read_jsonl(path):
                if any(d not in manifest["documents"] for d in row["retrieved_doc_ids"]):
                    raise ValueError(f"{row['id']}: document absent from the cache")
                annotated = annotate_row(row, corpus, tokenizer, manifest,
                                         cache_digest=digest, retained_end_cache=retained_ends)
                json.dump(annotated, handle, ensure_ascii=False)
                handle.write("\n")
                a = annotated["support_annotation"]
                positive = [v for y, v in zip(a["labels"], a["visible"]) if y]
                stats["questions"] += 1
                stats["positive_documents"] += len(positive)
                stats["visible_positive_documents"] += sum(v is True for v in positive)
                stats["unknown_positive_documents"] += sum(v is None for v in positive)
                stats["all_positive_visible_questions"] += int(bool(positive) and all(v is True for v in positive))
                stats["no_visible_positive_questions"] += int(not any(v is True for v in positive))
        report["splits"][name] = dict(stats)
    with open(os.path.join(args.out_dir, "visibility_report.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
