#!/usr/bin/env bash
# One arm per process. The caller selects a GPU/tmux session.
set -euo pipefail

crossdoc_phase=${1:-}
crossdoc_arm=${2:-}
crossdoc_seed=${3:-}
case "$crossdoc_phase" in train|eval) ;; *) echo "Usage: bash scripts/run_cross_document_validation.sh train|eval S0|SQ|S0X|SQX SEED [--dry-run]" >&2; exit 2;; esac
case "$crossdoc_arm" in S0|SQ|S0X|SQX) ;; *) echo "Unknown arm: $crossdoc_arm" >&2; exit 2;; esac
[[ "$crossdoc_seed" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 2; }
shift 3
crossdoc_dry=false
if [[ ${1:-} == --dry-run ]]; then crossdoc_dry=true; shift; fi
[[ $# == 0 ]] || { echo "Unexpected arguments: $*" >&2; exit 2; }
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

crossdoc_root=${CROSSDOC_RUN_ROOT:-/data02/quro/runs/crossdoc_validation_v1}
crossdoc_data=${CROSSDOC_DATA_DIR:-/data02/quro/data/hotpot}
crossdoc_cache=${CROSSDOC_CACHE_DIR:-/data02/quro/cache/hotpot-pisco-r16}
crossdoc_model=${CROSSDOC_PISCO_PATH:-/data02/quro/models/pisco-mistral}
crossdoc_steps=${CROSSDOC_STEPS:-3000}
crossdoc_run="$crossdoc_root/seed$crossdoc_seed/$crossdoc_arm"
crossdoc_cmd=(python -m src.train --preset pisco_shared_projector
  --seed "$crossdoc_seed" --data_order_seed "$crossdoc_seed"
  --projector_fusion none --support_loss_weight 0 --max_docs 10 --max_query_len 256
  --train_file "$crossdoc_data/train.jsonl" --cache_dir "$crossdoc_cache"
  --generator_path "$crossdoc_model" --num_workers 4
  --tag "crossdoc_${crossdoc_arm}_s${crossdoc_seed}")
if [[ $crossdoc_arm == S0* ]]; then
  crossdoc_cmd+=(--projector_query_mode none)
else
  crossdoc_cmd+=(--projector_query_mode conditioned)
fi
if [[ $crossdoc_arm == *X ]]; then crossdoc_cmd+=(--projector_cross_document); fi
if [[ $crossdoc_phase == train ]]; then
  crossdoc_output="$crossdoc_run"
  crossdoc_cmd+=(--steps "$crossdoc_steps" --lr 5e-5 --batch_size 2 --grad_accum 8
    --select_metric em --eval_every 250 --eval_every_samples 500
    --eval_files "dev=$crossdoc_data/dev.jsonl" --eval_max_samples 2000)
else
  crossdoc_split=${CROSSDOC_SPLIT:-test}
  case "$crossdoc_split" in dev|test|trivia) ;; *) echo "CROSSDOC_SPLIT must be dev, test or trivia" >&2; exit 2;; esac
  crossdoc_checkpoint=${CROSSDOC_CHECKPOINT:-$crossdoc_run/checkpoint_last.pt}
  crossdoc_output="$crossdoc_run/eval_$crossdoc_split"
  crossdoc_eval_file="$crossdoc_data/$crossdoc_split.jsonl"
  if [[ $crossdoc_split == trivia ]]; then
    # Out-of-domain: the HotpotQA-trained checkpoint reads the TriviaQA cache.
    # The train file is still loaded by build_loaders but never collated.
    crossdoc_eval_file=${CROSSDOC_TRIVIA_FILE:-/data02/quro/data/trivia/queries.jsonl}
    crossdoc_cache=${CROSSDOC_TRIVIA_CACHE:-/data02/quro/cache/trivia-pisco-r16}
    crossdoc_cmd+=(--cache_dir "$crossdoc_cache" --doc_control)
  fi
  crossdoc_cmd+=(--eval_only --resume_from "$crossdoc_checkpoint"
    --eval_files "$crossdoc_split=$crossdoc_eval_file" --eval_max_samples 999999)
  if [[ ${CROSSDOC_QUERY_CONTROL:-0} == 1 ]]; then
    crossdoc_output="${crossdoc_output}_query_control"
    crossdoc_cmd+=(--query_control)
  fi
fi
crossdoc_cmd+=(--out_dir "$crossdoc_output")
if $crossdoc_dry; then printf '%q ' "${crossdoc_cmd[@]}"; printf '\n'; exit 0; fi
for crossdoc_file in "$crossdoc_data/train.jsonl" "$crossdoc_cache/manifest.json"; do
  [[ -f $crossdoc_file ]] || { echo "Missing input: $crossdoc_file" >&2; exit 1; }
done
[[ -d $crossdoc_model ]] || { echo "Missing model directory: $crossdoc_model" >&2; exit 1; }
if [[ $crossdoc_phase == train ]]; then
  [[ -f $crossdoc_data/dev.jsonl ]] || { echo "Missing dev.jsonl" >&2; exit 1; }
else
  [[ -f $crossdoc_checkpoint && -f $crossdoc_eval_file ]] || { echo "Missing checkpoint or evaluation split" >&2; exit 1; }
fi
[[ ! -e $crossdoc_output ]] || { echo "Output already exists: $crossdoc_output; choose a fresh root" >&2; exit 1; }
exec "${crossdoc_cmd[@]}"
