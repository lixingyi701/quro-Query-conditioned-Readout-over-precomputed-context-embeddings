#!/usr/bin/env bash
# QuRO go/no-go gate: is query-conditioned readout worth anything at a locked budget?
#
# Stage 1 answers "does query conditioning help at all", Stage 2 answers "can the
# soft tokens carry the question by themselves".  Pass/fail criteria are fixed in
# advance (see scripts/summarize.py) precisely so they cannot be renegotiated
# after seeing the numbers.
#
#   bash scripts/run_gonogo.sh 1     # arms A / C / S / P, one per GPU
#   bash scripts/run_gonogo.sh 2     # A / C trained with query-text dropout
#
# SUPERSEDED for the A arms -- use scripts/run_arms.sh.  The "A" launched below
# does not pass --no_cosine_prior, and cosine_prior defaults to True, so the query
# still reaches the readout through the cosine bias on the first block's attention
# logits.  Everything it produced is A1 (cosine-conditioned), not the
# query-agnostic control; C-minus-A from this script is the value of *learned*
# conditioning on top of cosine, not the value of query conditioning.  Kept
# unchanged so the historical runs stay reproducible.
# See docs/HANDOFF.md §3 W1.
#
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE="${1:-1}"
STEPS="${STEPS:-3000}"
BUDGET="${BUDGET:-8}"
RUNS="${QURO_RUNS_DIR:-/data02/quro/runs}"
COMMON=(--preset pisco_gonogo --steps "$STEPS" --budget "$BUDGET"
        --budget_buckets 4,8 --query_control --eval_budgets 4,8)

launch () {   # launch <gpu> <tag> <extra args...>
  local gpu="$1" tag="$2"; shift 2
  echo "[gpu $gpu] $tag"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m src.train "${COMMON[@]}" "$@" \
      --tag "$tag" --out_dir "$RUNS/$tag" > "$RUNS/$tag.log" 2>&1 &
}

mkdir -p "$RUNS"

if [ "$STAGE" = "1" ]; then
  # C is the method; A removes the query from the output queries *only* -- the
  # cosine prior stays on, so this A is A1.  See the header.
  launch 0 gonogo_C_D0 --readout quro --output_query_mode xattn    --eval_input_modes D0
  launch 1 gonogo_A_D0 --readout quro --output_query_mode agnostic --eval_input_modes D0
  # S is query-conditioned but untrained: if the learned readout cannot beat
  # cosine similarity, it has not learned selection.
  launch 2 gonogo_S_D0 --readout similarity_topb --eval_input_modes D0
  # P is PISCO itself: not budget-matched (it emits K*m tokens), a reference row.
  launch 3 gonogo_P_D0 --readout pisco_direct --eval_input_modes D0
else
  # Training with the question text dropped half the time forces the readout to
  # carry query information, and makes the D1 evaluation in-distribution.
  launch 0 gonogo_C_qdrop --readout quro --output_query_mode xattn \
         --query_text_dropout 0.5 --eval_input_modes D0,D1
  launch 1 gonogo_A_qdrop --readout quro --output_query_mode agnostic \
         --query_text_dropout 0.5 --eval_input_modes D0,D1
  launch 2 gonogo_C_noqdrop --readout quro --output_query_mode xattn \
         --query_text_dropout 0.0 --eval_input_modes D0,D1
  launch 3 gonogo_S_qdrop --readout similarity_topb \
         --query_text_dropout 0.5 --eval_input_modes D0,D1
fi

wait
echo "stage $STAGE done"
python scripts/summarize.py --runs "$RUNS"
