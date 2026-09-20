#!/usr/bin/env bash
# Zero-shot transfer: score HotpotQA-trained arms on TriviaQA without retraining.
#
# TriviaQA's latents live in the gonogo cache, which was built with the settings
# the hotpot caches mirror -- same checkpoint, rate 16, m=8, float16,
# doc_max_length 128 -- so the vectors are interchangeable and a cache swap
# cannot be mistaken for a transfer effect. Verified before use: all 2534
# TriviaQA documents are present and both manifests agree on every protocol field.
#
# --doc_control is not optional here. A 7B answers most of TriviaQA from
# parametric memory, so a 70% score is mostly not evidence use; without the
# wrong-document floor the numbers cannot be read at all.
#
#   bash scripts/run_trivia_transfer.sh                    # the seven default arms
#   TAGS="hp2d0_P residual_ext_joint_s42" bash scripts/...  # a subset
set -euo pipefail
cd "$(dirname "$0")/.."

runs="${QURO_RUNS_DIR:-/data02/quro/runs}"
cache="${TRIVIA_CACHE:-/data02/quro/cache/gonogo-pisco-r16}"
queries="${TRIVIA_QUERIES:-/data02/quro/data/trivia/queries.jsonl}"
suffix="${SUFFIX:-_trivia}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3}"
tags="${TAGS:-hp2d0_P \
  residual_ext_joint_s42 residual_ext_joint_s43 residual_ext_joint_s44 \
  residual_ext_p-control_s42 residual_ext_p-control_s43 residual_ext_p-control_s44}"

for tag in $tags; do
  src="$runs/$tag"
  [ -f "$src/config.json" ] || { echo "missing $src/config.json" >&2; exit 1; }
  [ -f "$src/checkpoint_last.pt" ] || { echo "missing $src/checkpoint_last.pt" >&2; exit 1; }
  [ -e "$runs/${tag}${suffix}" ] && { echo "refusing to reuse $runs/${tag}${suffix}" >&2; exit 1; }
done

job() {
  tag="$1"; gpu="$2"
  src="/data02/quro/runs/$tag"; out="/data02/quro/runs/${tag}${SUFFIX_X}"
  # Each run is evaluated at its own budget. The label is not a truncation for
  # these arms -- P at B=8 and R at B=80 both hand the decoder every cached
  # latent, and in round one they agreed per-question 2000/2000 -- but a budget
  # outside a run's own buckets is rejected outright.
  budget=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['readout']['max_budget'])" "$src/config.json")
  mkdir -p "$out"
  git rev-parse HEAD > "$out/commit.txt"
  if ! CUDA_VISIBLE_DEVICES="$gpu" python -m src.train \
      --config_json "$src/config.json" --resume_from "$src/checkpoint_last.pt" \
      --out_dir "$out" --tag "${tag}${SUFFIX_X}" \
      --eval_only --eval_input_modes D0 --eval_budgets "$budget" --eval_max_samples 2000 \
      --cache_dir "$CACHE_X" --eval_files "trivia=$QUERIES_X" --doc_control \
      > "$out/console.log" 2>&1; then
    # xargs reports its own exit status, not the child's, so a silent failure
    # here would leave a "done" line next to a directory with no result.json.
    echo "FAILED $tag (gpu $gpu) -- see $out/console.log" >&2
    return 1
  fi
  echo "done $tag at B=$budget (gpu $gpu)"
}
export -f job
export SUFFIX_X="$suffix" CACHE_X="$cache" QUERIES_X="$queries"

i=0
for tag in $tags; do
  echo "$tag ${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  i=$((i + 1))
done | xargs -P "${#GPU_LIST[@]}" -n 2 bash -c 'job "$0" "$1"'
echo "all transfer evaluations finished"
