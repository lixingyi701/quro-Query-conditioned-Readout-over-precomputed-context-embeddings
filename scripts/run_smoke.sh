#!/usr/bin/env bash
# Level 0: prove the whole chain works before spending hours on the real run.
#
# Builds a 4k-document cache, trains briefly, and checks that a QuRO degenerated
# into PISCO reproduces PISCO's own answers -- which validates the cache
# round-trip, the prompt template, the slot indexing and the embedding injection
# all at once.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${GPU:-0}"
DATA="${QURO_DATA_DIR:-/data02/quro/data}/smoke"
CACHE="${QURO_CACHE_DIR:-/data02/quro/cache}/smoke-pisco-r16"
RUNS="${QURO_RUNS_DIR:-/data02/quro/runs}"

echo "== 0. contract tests (CPU) =="
python tests/test_shapes.py

if [ ! -f "$DATA/train.jsonl" ]; then
  echo "== 1. prepare data =="
  python scripts/prepare_selecom_data.py --out_dir "$DATA" \
    --stage1_rows 2000 --stage2_rows 200 --stage2_scan 2000 --dev_fraction 0.05
fi

if [ ! -f "$CACHE/manifest.json" ]; then
  echo "== 2. build latent cache =="
  CUDA_VISIBLE_DEVICES="$GPU" python scripts/build_latent_cache.py \
    --documents "$DATA/corpus.jsonl" --out_dir "$CACHE" \
    --adapter src.compressors.pisco:build --batch_size 64 --dtype float16 --resume
fi

echo "== 3. QuRO == PISCO regression =="
CUDA_VISIBLE_DEVICES="$GPU" python scripts/check_pisco_equivalence.py \
  --rows 32 --out "$RUNS/pisco_equivalence.json"

echo "== 4. short training run =="
CUDA_VISIBLE_DEVICES="$GPU" python -m src.train --preset pisco_smoke \
  --steps 200 --tag smoke_C --out_dir "$RUNS/smoke_C" \
  --query_control --eval_budgets 4,8 --eval_input_modes D0,D1

echo "smoke complete: $RUNS/smoke_C/result.json"
