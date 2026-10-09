#!/bin/bash
# Wait for all 43 kilt-128 shards, then encode (4 GPUs) -> search (1 GPU) -> rerank (4 GPUs).
set -euo pipefail
cd /home/lxy/quro
W=/data02/quro/data/public_qa_90k_retrieval
BLOBS=/data02/quro/hf-cache/datasets--dmrau--kilt-128/blobs
log() { echo "[$(date +%T)] $*" | tee -a $W/run_pipeline.log; }

until [ "$(ls $BLOBS/shard_*.parquet 2>/dev/null | wc -l)" -eq 43 ] && ! ls $BLOBS/shard_*.tmp >/dev/null 2>&1; do
  sleep 30
done
log "all 43 shards present"
python - <<'PY' || { log "parquet integrity check failed"; exit 1; }
import glob, pyarrow.parquet as pq
paths = sorted(glob.glob("/data02/quro/hf-cache/datasets--dmrau--kilt-128/blobs/shard_*.parquet"))
total = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)
print(f"{len(paths)} shards, {total} rows")
assert len(paths) == 43 and total > 20_000_000
PY
log "parquet integrity ok"

[ -f $W/search.npz ] && [ -f $W/retrieval.jsonl ] && { log "retrieval already complete"; exit 0; }
log "encode start"
pids=()
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r python scripts/build_splade_retrieval.py encode --gpu_rank $r --world 4 \
    > $W/encode_rank$r.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
[ "$(ls $W/enc/shard_*.npz | wc -l)" -eq 43 ] || { log "encode incomplete"; exit 1; }
log "encode done"

log "search start"
CUDA_VISIBLE_DEVICES=0 python scripts/build_splade_retrieval.py search > $W/search.log 2>&1
log "search done"

log "rerank start"
pids=()
for r in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$r python scripts/build_splade_retrieval.py rerank --gpu_rank $r --world 4 \
    > $W/rerank_rank$r.log 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p; done
cat $W/rerank_rank{0,1,2,3}.jsonl > $W/retrieval.jsonl
log "rerank done: $(wc -l < $W/retrieval.jsonl) queries -> $W/retrieval.jsonl"
