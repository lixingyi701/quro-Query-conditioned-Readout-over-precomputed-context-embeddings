"""Import an existing ``.pt`` latent export into QuRO's sharded cache.

Accepted payloads:
  * ``{doc_id: tensor(m,h)}``
  * ``{'doc_ids': [...], 'latents': tensor(n,m,h), 'source_token_counts': [...]}``
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache import import_tensor_records


def load_payload(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def records(payload):
    if "doc_ids" in payload and "latents" in payload:
        counts = payload.get("source_token_counts", [0] * len(payload["doc_ids"]))
        for doc_id, latent, count in zip(payload["doc_ids"], payload["latents"], counts):
            yield str(doc_id), latent, int(count)
        return
    for doc_id, latent in payload.items():
        if not torch.is_tensor(latent):
            raise TypeError("mapping payload values must all be tensors")
        yield str(doc_id), latent, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--compressor", required=True, help="checkpoint/model identifier recorded in manifest")
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--shard_size", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    path = import_tensor_records(records(load_payload(args.input)), args.out_dir, args.compressor,
                                 dtype=args.dtype, shard_size=args.shard_size,
                                 overwrite=args.overwrite)
    print(path)


if __name__ == "__main__":
    main()
