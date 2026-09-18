#!/usr/bin/env bash
# Starts from the actual trained P run, with its data/cache/backbone configuration.
# No launch by default: the caller explicitly chooses init, train, or controls.
set -euo pipefail
cd "$(dirname "$0")/.."

stage="${1:?usage: bash scripts/run_pisco_residual.sh init|train|joint|p-control}"
baseline="${BASELINE_RUN:-/data02/quro/runs/hp2d0_P}"
runs="${QURO_RUNS_DIR:-/data02/quro/runs}"
seed="${SEED:-42}"
tag="${TAG:-residual_${stage}_s${seed}}"
out="$runs/$tag"
if [ -e "$out" ]; then
  echo "Refusing to reuse $out; choose a fresh TAG." >&2
  exit 1
fi
flags=()
case "$stage" in
  init) flags=(--arm R --eval_only --generator_lora_init frozen) ;;
  train) flags=(--arm R --generator_lora_init frozen) ;;
  joint) flags=(--arm R --generator_lora_init pisco) ;;
  p-control) flags=(--arm P --generator_lora_init pisco) ;;
  *) echo "unknown stage: $stage" >&2; exit 1 ;;
esac
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
  --eval_max_samples 2000 "${flags[@]}" \
  2>&1 | tee "$out/console.log"
