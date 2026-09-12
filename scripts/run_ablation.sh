#!/usr/bin/env bash
# A (agnostic) vs C/D (concat/xattn) under the same cache, seed, K and B.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CACHE_DIR:?set CACHE_DIR}"
: "${RAG_TRAIN_FILE:?set RAG_TRAIN_FILE}"
: "${RAG_DEV_FILE:?set RAG_DEV_FILE}"
: "${ENCODER_PATH:?set ENCODER_PATH to the query encoder}"
: "${GENERATOR_PATH:?set GENERATOR_PATH to the causal LM}"

ROOT=${ROOT:-runs/ablation_v0}
STEPS=${STEPS:-1500}
BS_LIST=${BS_LIST:-"4 8 16 32"}
for B in $BS_LIST; do
  for MODE in agnostic concat xattn; do
    python3 -m src.train --preset qwen3emb --cache_dir "$CACHE_DIR" \
      --encoder "$ENCODER_PATH" --generator "$GENERATOR_PATH" \
      --train_file "$RAG_TRAIN_FILE" --eval_files "dev=$RAG_DEV_FILE" \
      --max_docs "${MAX_DOCS:-5}" --num_compressed "$B" --budget_buckets "$B" \
      --output_query_mode "$MODE" --steps "$STEPS" --seed "${SEED:-42}" \
      --generator_lora --bf16 --tag "B${B}-${MODE}" \
      --out_dir "$ROOT/B${B}_${MODE}"
  done
done
python3 scripts/summarize.py "$ROOT"/B*_*
