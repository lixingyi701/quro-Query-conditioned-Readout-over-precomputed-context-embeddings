#!/usr/bin/env bash
# Phase 1-2 of the SeleCom infeasibility replication, one run per GPU.
#
# Three configurations, not one, because a null result under PISCO's own prompt
# would otherwise be unattributable: the system prompt already orders the model
# to extract from the provided documents, and the `Question:` marker tells a
# RAG-tuned decoder it is being asked about the background.  Both are pulls
# towards the document that have nothing to do with compression, so the same
# Level A set is also run with neither (`selecom_literal`, no system prompt).
#
#   usage: bash scripts/run_full_compression_infeasibility.sh [rows_A] [rows_B]
set -euo pipefail

ROOT="${QURO_ROOT:-/data02/quro}/results/full_compression_infeasibility"
ROWS_A="${1:-40}"
ROWS_B="${2:-200}"
mkdir -p "$ROOT"

cd "$(dirname "$0")/.."

launch() {  # launch <gpu> <run_id> <args...>
  local gpu="$1" run_id="$2"; shift 2
  rm -rf "${ROOT:?}/${run_id}"
  echo "[gpu $gpu] $run_id"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -u \
    scripts/diagnose_full_compression_infeasibility.py --run_id "$run_id" "$@" \
    > "$ROOT/${run_id}.log" 2>&1 &
}

launch 0 levelA-pisco-prompt   --level A --rows "$ROWS_A" --figure_samples 3
launch 1 levelA-selecom-prompt --level A --rows "$ROWS_A" --figure_samples 3 \
        --prompt_style selecom_literal --system_prompt none
launch 2 levelB-pisco-prompt   --level B --rows "$ROWS_B" --figure_samples 3

wait
echo "all runs finished; logs under $ROOT"
