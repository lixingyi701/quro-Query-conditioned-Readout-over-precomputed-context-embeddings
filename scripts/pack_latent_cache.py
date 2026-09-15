"""Repack a sharded latent cache into one memory-mapped file.

Shards are the right format for *building* a cache -- incremental, resumable,
crash-safe.  They are the wrong format for *training*: batches are shuffled, so
one batch of 4 queries at K=10 touches up to 40 documents spread across every
shard, and an LRU over half-gigabyte shards reloads instead of caching.  Measured
on the 264k-document cache, training stalled at 0% GPU utilisation.

After packing, a document read is one page fault into ``latents.bin``, and
because the OS page cache is shared between processes, four training arms running
in parallel share a single 17 GB resident copy instead of holding one each.

    python scripts/pack_latent_cache.py --cache /data02/quro/cache/gonogo-pisco-r16

The shards are kept unless ``--remove_shards`` is passed, so the packed cache can
always be rebuilt and nothing is destroyed by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache import LATENT_BIN, _safe_torch_load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--remove_shards", action="store_true",
                    help="delete the .pt shards after verifying the packed file")
    args = ap.parse_args()

    root = os.path.abspath(args.cache)
    manifest_path = os.path.join(root, "manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("storage") == "memmap":
        print(f"{root} is already packed")
        return

    documents = manifest["documents"]
    shards = manifest["shards"]
    m, h = int(manifest["latent_size"]), int(manifest["hidden_size"])
    dtype = np.dtype(manifest["dtype"])
    total = len(documents)
    print(f"packing {total} documents from {len(shards)} shards "
          f"-> ({total}, {m}, {h}) {dtype.name} "
          f"= {total * m * h * dtype.itemsize / 1e9:.1f} GB")

    # Row order is by shard then row, so each shard is read exactly once.
    by_shard: dict[int, list[tuple[int, str]]] = {}
    for doc_id, loc in documents.items():
        by_shard.setdefault(int(loc["shard"]), []).append((int(loc["row"]), doc_id))

    bin_path = os.path.join(root, LATENT_BIN)
    tmp_path = bin_path + ".tmp"
    packed = np.memmap(tmp_path, dtype=dtype, mode="w+", shape=(total, m, h))
    index, started = 0, time.time()
    for shard_idx in sorted(by_shard):
        payload = _safe_torch_load(os.path.join(root, str(shards[shard_idx]["file"])),
                                   map_location="cpu")
        latents = payload["latents"].numpy()
        for row, doc_id in sorted(by_shard[shard_idx]):
            packed[index] = latents[row]
            documents[doc_id]["index"] = index
            index += 1
        del payload, latents
        print(f"  shard {shard_idx + 1}/{len(shards)}  {index}/{total} rows  "
              f"{time.time() - started:.0f}s", flush=True)
    packed.flush()
    del packed
    os.replace(tmp_path, bin_path)

    if index != total:
        raise RuntimeError(f"packed {index} rows but the manifest lists {total}")

    manifest["storage"] = "memmap"
    manifest["documents"] = documents
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(tmp, manifest_path)

    # Read back through the public interface before touching the originals.
    from src.cache import LatentCache
    cache = LatentCache(root)
    assert cache.storage == "memmap", cache.storage
    sample = list(documents)[: min(64, total)]
    latents, mask, counts = cache.get_many([[doc_id] for doc_id in sample])
    if not np.isfinite(latents.float().numpy()).all():
        raise RuntimeError("packed cache contains NaN or Inf")
    print(f"verified {len(sample)} documents through LatentCache: {tuple(latents.shape)}")

    if args.remove_shards:
        for shard in shards:
            os.remove(os.path.join(root, str(shard["file"])))
        print(f"removed {len(shards)} shards")
    else:
        print(f"kept {len(shards)} shards (pass --remove_shards to reclaim "
              f"{sum(int(s['count']) for s in shards) * m * h * dtype.itemsize / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
