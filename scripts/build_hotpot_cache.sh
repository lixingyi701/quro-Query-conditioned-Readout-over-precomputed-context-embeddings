#!/usr/bin/env bash
# Compress HotpotQA's paragraphs once, in four parallel slices, then merge.
#
# Settings mirror /data02/quro/cache/gonogo-pisco-r16/manifest.json exactly --
# same checkpoint, rate 16, m=8, float16, doc_max_length 128 -- so the two caches
# are interchangeable and a compressor change cannot be mistaken for a dataset
# effect.
#
#   bash scripts/build_hotpot_cache.sh          # build all four slices, then merge
#   MERGE_ONLY=1 bash scripts/build_hotpot_cache.sh
#
# The merge is the dangerous step, not the build. A previous merge mis-mapped the
# index and handed back another document's latents for half the corpus, with
# correct shapes, no NaNs and a passing read-back check (HANDOFF §7). What caught
# it in the end, and what pack_latent_cache.py now runs unconditionally, is a
# shard-level round trip: for each source directory it re-reads probe documents
# out of the shard they claim to come from and compares them byte for byte
# against the packed row. Do not skip it, and do not pass --remove_shards until
# it has printed its count.
set -euo pipefail
cd "$(dirname "$0")/.."

CORPUS="${CORPUS:-/data02/quro/data/hotpot/corpus.jsonl}"
CACHE="${CACHE:-/data02/quro/cache/hotpot-pisco-r16}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3}"
N="${#GPU_LIST[@]}"

TOTAL=$(wc -l < "$CORPUS")
SLICE=$(( (TOTAL + N - 1) / N ))
echo "corpus: $TOTAL paragraphs -> $N slices of $SLICE on GPUs ${GPU_LIST[*]}"

if [ -z "${MERGE_ONLY:-}" ]; then
  for i in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$i]}"
    offset=$(( i * SLICE ))
    out="${CACHE}-part${i}"
    echo "[gpu $gpu] offset=$offset limit=$SLICE -> $out"
    CUDA_VISIBLE_DEVICES="$gpu" nohup python scripts/build_latent_cache.py \
        --documents "$CORPUS" --out_dir "$out" \
        --offset "$offset" --limit "$SLICE" \
        --adapter src.compressors.pisco:build \
        --checkpoint /data02/quro/models/pisco-mistral \
        --batch_size 64 --shard_size 8192 --dtype float16 \
        > "${CACHE}-part${i}.log" 2>&1 &
  done
  wait
  echo "all slices built"
fi

# part0 becomes the merged cache; the rest fold into it.  Keep the shards until
# the round-trip check has passed -- they are the only copy of the ground truth
# the check compares against.
EXTRA=()
for i in $(seq 1 $((N - 1))); do EXTRA+=("${CACHE}-part${i}"); done
python scripts/pack_latent_cache.py --cache "${CACHE}-part0" "${EXTRA[@]:+--merge}" "${EXTRA[@]:+${EXTRA[@]}}"

# The preset points at $CACHE, so expose the merged part0 under that name.
if [ ! -e "$CACHE" ]; then ln -s "${CACHE}-part0" "$CACHE"; fi

python - "$CACHE" "$CORPUS" <<'EOF'
# Independent of the packer's own probes: sample across the whole corpus and
# confirm every document resolves, is finite, and is not a duplicate of its
# neighbour -- an off-by-one merge produces exactly that signature.
import json, sys, torch
sys.path.insert(0, ".")
from src.cache import LatentCache

cache_dir, corpus_path = sys.argv[1], sys.argv[2]
cache = LatentCache(cache_dir)
ids = [json.loads(l)["doc_id"] for l in open(corpus_path, encoding="utf-8")]
print(f"cache holds {len(cache)} documents, corpus file has {len(ids)}")
missing = [d for d in ids[:: max(1, len(ids) // 500)] if d not in cache]
if missing:
    raise SystemExit(f"{len(missing)} sampled documents are absent from the cache")
probe = ids[:: max(1, len(ids) // 200)][:200]
vecs = torch.stack([cache.get(d)[0].float().flatten() for d in probe])
if not torch.isfinite(vecs).all():
    raise SystemExit("cache returned NaN or Inf")
dup = sum(1 for i in range(len(probe) - 1) if torch.equal(vecs[i], vecs[i + 1]))
print(f"checked {len(probe)} documents, {dup} adjacent-identical (expect 0)")
if dup:
    raise SystemExit("adjacent documents share latents; the merge mapping is suspect")
print("cache OK")
EOF
