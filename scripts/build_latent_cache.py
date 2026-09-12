"""Build a QuRO cache with a frozen external PISCO/COCOM adapter.

Input JSONL rows require ``doc_id`` and either ``text`` or ``document``.
The adapter factory contract is documented in ``src.offline.load_adapter``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache import CacheMetadata, LatentCacheWriter
from src.offline import load_adapter


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", required=True, help="JSONL with doc_id and text/document")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--adapter", required=True, help="module.path:factory")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--shard_size", type=int, default=1024)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with open(args.documents, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError("document file is empty")
    seen = set()
    for row in rows:
        row["doc_id"] = str(row["doc_id"])
        row["text"] = str(row.get("text", row.get("document", "")))
        if not row["doc_id"] or row["doc_id"] in seen:
            raise ValueError(f"empty or duplicate doc_id: {row['doc_id']!r}")
        seen.add(row["doc_id"])

    adapter = load_adapter(args.adapter, args.checkpoint, args.device)
    writer = None
    try:
        done = 0
        for group in batches(rows, args.batch_size):
            values = adapter.encode_texts([row["text"] for row in group])
            if writer is None:
                metadata = CacheMetadata(
                    compressor=f"{adapter.name}:{args.checkpoint or 'default'}",
                    latent_size=values.size(1),
                    hidden_size=values.size(2),
                    dtype=args.dtype,
                )
                writer = LatentCacheWriter(
                    args.out_dir, metadata, shard_size=args.shard_size, overwrite=args.overwrite)
            for row, latent in zip(group, values):
                n_tokens = int(row.get("source_token_count", 0))
                writer.add(row["doc_id"], latent, n_tokens)
            done += len(group)
            print(f"cached {done}/{len(rows)} documents")
        manifest = writer.close()
        print(f"saved cache manifest: {manifest}")
    except Exception:
        raise


if __name__ == "__main__":
    main()
