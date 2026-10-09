#!/usr/bin/env bash
# Reuse the audited Wikipedia index, search new queries, rerank, validate merge.
# Run inside tmux. WORK_DIR is required; KEEP=10 builds prefix-compatible eval K5/K10.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${WORK_DIR:?set WORK_DIR to a new query-specific output directory}"
INDEX_DIR="${INDEX_DIR:-$WORK_DIR/enc}"
KEEP="${KEEP:-5}"
DEPTH="${DEPTH:-50}"
QUERY_BLOCK="${QUERY_BLOCK:-128}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3}"
WORLD="${#GPU_LIST[@]}"
mkdir -p "$WORK_DIR"

wait_jobs() {
  local failed=0 pid
  for pid in "$@"; do if ! wait "$pid"; then failed=1; fi; done
  if (( failed )); then echo "A retrieval worker failed; inspect $WORK_DIR/*.log" >&2; return 1; fi
}

pids=()
for rank in "${!GPU_LIST[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$rank]}" python scripts/build_splade_retrieval.py encode \
    --work_dir "$WORK_DIR" --index_dir "$INDEX_DIR" --gpu_rank "$rank" --world "$WORLD" "$@" \
    > "$WORK_DIR/encode_rank$rank.log" 2>&1 &
  pids+=("$!")
done
wait_jobs "${pids[@]}"
CUDA_VISIBLE_DEVICES="${GPU_LIST[0]}" python scripts/build_splade_retrieval.py search \
  --work_dir "$WORK_DIR" --index_dir "$INDEX_DIR" --depth "$DEPTH" --query_block "$QUERY_BLOCK" "$@" \
  > "$WORK_DIR/search.log" 2>&1
pids=()
for rank in "${!GPU_LIST[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU_LIST[$rank]}" python scripts/build_splade_retrieval.py rerank \
    --work_dir "$WORK_DIR" --index_dir "$INDEX_DIR" --gpu_rank "$rank" --world "$WORLD" --keep "$KEEP" "$@" \
    > "$WORK_DIR/rerank_rank$rank.log" 2>&1 &
  pids+=("$!")
done
wait_jobs "${pids[@]}"
python - "$WORK_DIR" "$WORLD" "$KEEP" <<'PY'
import json, os, sys
from pathlib import Path
import numpy as np
root, world, keep = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
expected = set(np.load(root / "search.npz")["ids"].tolist())
seen = set()
with (root / "retrieval.jsonl.tmp").open("w") as output:
    for rank in range(world):
        with (root / f"rerank_rank{rank}.jsonl").open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                docs = row.get("documents", [])
                if row["id"] in seen or len(docs) != keep or len({d["doc_id"] for d in docs}) != keep:
                    raise SystemExit("duplicate question or incomplete/distinct-document ranking")
                seen.add(row["id"])
                output.write(line)
if seen != expected:
    raise SystemExit(f"retrieval query coverage differs: expected {len(expected)}, got {len(seen)}")
os.replace(root / "retrieval.jsonl.tmp", root / "retrieval.jsonl")
print(f"retrieval complete: {len(seen)} questions, K={keep}, {root / 'retrieval.jsonl'}")
PY
