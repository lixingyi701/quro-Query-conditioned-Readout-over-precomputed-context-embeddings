#!/usr/bin/env bash
# One arm per process/GPU; caller owns tmux and CUDA_VISIBLE_DEVICES.
set -euo pipefail

task_fusion_mode=${1:-}
case "$task_fusion_mode" in
  additive|film) shift ;;
  *) echo "Usage: bash scripts/run_projector_fusion.sh additive|film [--dry-run]" >&2; exit 2 ;;
esac
task_fusion_dry_run=false
if [[ ${1:-} == --dry-run ]]; then task_fusion_dry_run=true; shift; fi
if [[ $# -ne 0 ]]; then echo "Unexpected arguments: $*" >&2; exit 2; fi

task_fusion_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd -- "$task_fusion_repo"
task_fusion_sq=${SQ_CHECKPOINT:-/data02/quro/runs/shared_projector_v1/shared_contextual/checkpoint_last.pt}
task_fusion_root=${FUSION_RUN_ROOT:-/data02/quro/runs/query_fusion_v1}
task_fusion_data=${SUPPORT_DATA_DIR:-/data02/quro/data/hotpot-support-visible}
task_fusion_output="$task_fusion_root/$task_fusion_mode"
task_fusion_support_weight=${SUPPORT_LOSS_WEIGHT:-0.1}
task_fusion_support_warmup=${SUPPORT_WARMUP_STEPS:-100}
task_fusion_command=(
  python -m src.train --preset pisco_shared_projector --projector_fusion "$task_fusion_mode"
  --support_head --support_head_input output --support_loss_weight "$task_fusion_support_weight"
  --support_warmup_steps "$task_fusion_support_warmup"
  --resume_from "$task_fusion_sq" --warm_start
  --steps 1000 --lr 2e-5 --seed 42 --batch_size 2 --grad_accum 8
  --select_metric f1 --eval_every 100 --eval_every_samples 500 --eval_max_samples 2000
  --train_file "$task_fusion_data/train.jsonl" --eval_files "dev=$task_fusion_data/dev.jsonl"
  --support_visibility_policy visible --query_control --doc_control --out_dir "$task_fusion_output"
)
if $task_fusion_dry_run; then
  printf '%q ' "${task_fusion_command[@]}"
  printf '\n'
  exit 0
fi
for task_fusion_input in "$task_fusion_sq" "$task_fusion_data/train.jsonl" "$task_fusion_data/dev.jsonl"; do
  if [[ ! -f "$task_fusion_input" ]]; then echo "Missing input: $task_fusion_input" >&2; exit 1; fi
done
if [[ -e "$task_fusion_output/train_log.jsonl" || -e "$task_fusion_output/checkpoint_last.pt" ]]; then
  echo "Run already exists: $task_fusion_output; choose a fresh FUSION_RUN_ROOT or resume explicitly." >&2
  exit 1
fi
exec "${task_fusion_command[@]}"
