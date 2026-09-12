#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CACHE_DIR:?set CACHE_DIR to a PISCO/COCOM latent cache}"
: "${RAG_TRAIN_FILE:?set RAG_TRAIN_FILE}"
: "${RAG_DEV_FILE:?set RAG_DEV_FILE}"

MODEL=${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}
OUT=${OUT:-runs/quro_mistral7b}
python3 -m src.train --preset qwen7b --cache_dir "$CACHE_DIR" \
  --generator "$MODEL" --generator_lora \
  --train_file "$RAG_TRAIN_FILE" --eval_files "dev=$RAG_DEV_FILE" \
  --max_docs "${MAX_DOCS:-5}" --num_compressed "${BUDGET:-40}" \
  --budget_buckets "${BUDGET:-40}" --steps "${STEPS:-3000}" \
  --output_query_mode xattn --query_control --bf16 --out_dir "$OUT"
