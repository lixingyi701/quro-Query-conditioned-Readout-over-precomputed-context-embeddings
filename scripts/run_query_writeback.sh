#!/usr/bin/env bash
# RQ variant: query-as-Q reads latent KV, then A^T writes back. R remains separate.
# No launch by default: the caller explicitly chooses init, train, or controls.
set -euo pipefail
cd "$(dirname "$0")/.."

stage="${1:?usage: bash scripts/run_query_writeback.sh init|train|joint|p-control}"
baseline="${BASELINE_RUN:-/data02/quro/runs/hp2d0_P}"
runs="${QURO_RUNS_DIR:-/data02/quro/runs}"
seed="${SEED:-42}"
tag="${TAG:-query_writeback_${stage}_s${seed}}"
out="$runs/$tag"
if [ -e "$out" ]; then
  echo "Refusing to reuse $out; choose a fresh TAG." >&2
  exit 1
fi
flags=()
case "$stage" in
  init) flags=(--arm RQ --eval_only --generator_lora_init frozen) ;;
  train) flags=(--arm RQ --generator_lora_init frozen) ;;
  joint) flags=(--arm RQ --generator_lora_init pisco) ;;
  p-control) flags=(--arm P --generator_lora_init pisco) ;;
  *) echo "unknown stage: $stage" >&2; exit 1 ;;
esac
# Training data is the one variable the residual results left open, so it is
# overridable while everything else still comes from the P run's config.  The
# eval files are deliberately NOT overridable: dev is the anchor every historical
# number is read against.  A CACHE_DIR must be a superset of the baseline's, or
# the same dev documents come back as different latents and the comparison to
# 54.50 is no longer a comparison.
data_flags=()
[ -n "${TRAIN_FILE:-}" ] && data_flags+=(--train_file "$TRAIN_FILE")
[ -n "${CACHE_DIR:-}" ] && data_flags+=(--cache_dir "$CACHE_DIR")
[ ${#data_flags[@]} -gt 0 ] && data_flags+=(--allow_data_change)
mkdir -p "$out"
git rev-parse HEAD > "$out/commit.txt"
git diff HEAD > "$out/worktree.diff"

python -m src.train \
  --baseline_run "$baseline" --out_dir "$out" --tag "$tag" \
  --budget 80 --budget_buckets 80 --eval_budgets 80 --no_budget_dropout \
  --decoder_input_mode D0 --query_text_dropout 0 --eval_input_modes D0 \
  --query_representation fixed_adapter --seed "$seed" \
  --d_readout "${D_READOUT:-256}" --readout_blocks 1 \
  --steps "${STEPS:-3000}" --lr "${LR:-0.0001}" \
  --eval_every 500 --eval_every_samples 500 --select_metric em \
  --eval_max_samples 2000 "${flags[@]}" "${data_flags[@]+"${data_flags[@]}"}" \
  2>&1 | tee "$out/console.log"
