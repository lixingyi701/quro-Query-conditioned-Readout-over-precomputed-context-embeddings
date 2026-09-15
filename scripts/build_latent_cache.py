"""Compress a corpus once with a frozen compressor and shard the latents to disk.

This is QuRO's offline/online boundary: everything expensive about a document
happens here, exactly once, and online readout never touches source tokens again.

    python scripts/build_latent_cache.py \
      --documents /data02/quro/data/gonogo/corpus.jsonl \
      --out_dir   /data02/quro/cache/pisco-r16 \
      --adapter   src.compressors.pisco:build

Input rows need ``doc_id`` and ``text``/``document``.  Token counts are taken
from the compressor itself rather than the input file, so the recorded
``source_token_count`` is what the compressor actually consumed after its own
truncation -- the only honest denominator for xi_off.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache import CacheMetadata, LatentCacheWriter
from src.offline import load_adapter


def batches(rows, size):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def read_documents(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise ValueError(f"document file is empty: {path}")
    seen = set()
    for row in rows:
        row["doc_id"] = str(row["doc_id"])
        row["text"] = str(row.get("text", row.get("document", "")))
        if not row["doc_id"] or row["doc_id"] in seen:
            raise ValueError(f"empty or duplicate doc_id: {row['doc_id']!r}")
        if not row["text"].strip():
            raise ValueError(f"empty document text for {row['doc_id']}")
        seen.add(row["doc_id"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--documents", required=True, help="JSONL with doc_id and text/document")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--adapter", default="src.compressors.pisco:build",
                    help="module.path:factory returning encode_texts -> (B,m,h)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--shard_size", type=int, default=4096)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--limit", type=int, default=None, help="cache only the first N documents")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N documents; with --limit this carves out a "
                         "contiguous slice so several GPUs can build one cache in "
                         "parallel, each into its own out_dir, merged afterwards by "
                         "scripts/pack_latent_cache.py --merge")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true", help="continue an interrupted cache")
    args = ap.parse_args()

    rows = read_documents(args.documents)
    if args.offset:
        rows = rows[args.offset:]
    if args.limit:
        rows = rows[: args.limit]

    adapter = load_adapter(args.adapter, args.checkpoint, args.device)
    backend = adapter.backend
    info = backend.metadata() if hasattr(backend, "metadata") else {}
    metadata = CacheMetadata(
        compressor=info.get("compressor", adapter.name),
        latent_size=int(info["latent_size"]),
        hidden_size=int(info["hidden_size"]),
        dtype=args.dtype,
        compr_rate=info.get("compr_rate"),
        doc_max_length=info.get("doc_max_length"),
        checkpoint=info.get("checkpoint", args.checkpoint),
    )
    print(f"[cache] {metadata.compressor}: m={metadata.latent_size} h={metadata.hidden_size} "
          f"rate={metadata.compr_rate} doc_max_length={metadata.doc_max_length}")

    writer = LatentCacheWriter(args.out_dir, metadata, shard_size=args.shard_size,
                               overwrite=args.overwrite, resume=args.resume)
    already = writer.cached_ids
    if already:
        rows = [row for row in rows if row["doc_id"] not in already]
        print(f"[cache] resuming: {len(already)} already cached, {len(rows)} to go")
    if not rows:
        print("[cache] nothing to do")
        writer.close()
        return

    token_stats = getattr(backend, "token_stats", None)
    started, done, fed_total, original_total = time.time(), 0, 0, 0
    for group in batches(rows, args.batch_size):
        texts = [row["text"] for row in group]
        latents = adapter.encode_texts(texts)
        stats = (token_stats(texts) if token_stats is not None
                 else [{"fed_tokens": int(row.get("source_token_count", 0)),
                        "original_tokens": int(row.get("source_token_count", 0))}
                       for row in group])
        for row, latent, stat in zip(group, latents, stats):
            writer.add(row["doc_id"], latent, stat["fed_tokens"], stat["original_tokens"])
            fed_total += stat["fed_tokens"]
            original_total += stat["original_tokens"]
        done += len(group)
        if done % (args.batch_size * 20) == 0 or done == len(rows):
            rate = done / max(1e-6, time.time() - started)
            eta = (len(rows) - done) / max(1e-6, rate)
            print(f"[cache] {done}/{len(rows)} docs  {rate:.0f} doc/s  eta {eta/60:.1f} min")

    manifest = writer.close()
    xi_off = fed_total / max(1, done * metadata.latent_size)
    print(f"[cache] saved {manifest}")
    print(f"[cache] mean fed tokens/doc {fed_total/max(1,done):.1f} "
          f"(untruncated {original_total/max(1,done):.1f}, "
          f"truncation loss {100*(1-fed_total/max(1,original_total)):.1f}%)")
    print(f"[cache] xi_off = fed_tokens / m = {xi_off:.2f}x")


if __name__ == "__main__":
    main()
