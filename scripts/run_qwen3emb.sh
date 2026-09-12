#!/usr/bin/env bash
# Cache-first QuRO: frozen PISCO/COCOM cache + query encoder + generator LoRA.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${CACHE_DIR:?set CACHE_DIR to a QuRO latent cache}"
: "${RAG_TRAIN_FILE:?set RAG_TRAIN_FILE}"
: "${RAG_DEV_FILE:?set RAG_DEV_FILE}"

ENCODER_PATH=${ENCODER_PATH:-/path/to/Qwen3-Embedding-0.6B}
GENERATOR_PATH=${GENERATOR_PATH:-/path/to/Qwen2.5-1.5B-Instruct}
OUT=${OUT:-runs/quro_v0}
STEPS=${STEPS:-3000}
BS=${BS:-4}
ACC=${ACC:-2}
MAX_DOCS=${MAX_DOCS:-5}
BUDGET=${BUDGET:-8}

COMMON=(--preset qwen3emb --cache_dir "$CACHE_DIR"
        --encoder "$ENCODER_PATH" --generator "$GENERATOR_PATH" --generator_lora
        --train_file "$RAG_TRAIN_FILE" --eval_files "dev=$RAG_DEV_FILE"
        --max_docs "$MAX_DOCS" --num_compressed "$BUDGET"
        --budget_buckets "$BUDGET" --batch_size "$BS" --grad_accum "$ACC"
        --steps "$STEPS" --bf16)

python3 -m src.train "${COMMON[@]}" --output_query_mode xattn \
  --query_control --tag "quro-xattn-B${BUDGET}" --out_dir "${OUT}_xattn"

python3 -m src.train "${COMMON[@]}" --output_query_mode agnostic \
  --tag "agnostic-B${BUDGET}" --out_dir "${OUT}_agnostic"

python3 scripts/summarize.py "${OUT}_xattn" "${OUT}_agnostic"
