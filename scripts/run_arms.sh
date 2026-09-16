#!/usr/bin/env bash
# The B=8 arm matrix from docs/warning_and_target.md §4.1.
#
# Why this exists alongside run_gonogo.sh: every A arm that has ever been run
# carried cosine_prior=True (audited over /data02/quro/runs/*/config.json on
# 2026-09-16).  The cosine prior is a second, independent route by which the
# question reaches the readout, so those runs are A1 -- "cosine-conditioned" --
# and not the query-agnostic control they were reported against.  The headline
# "C beats a query-agnostic readout by 20.85 EM in D1" therefore has no A0 behind
# it.  run_gonogo.sh is left untouched so the historical runs stay reproducible;
# this script launches the arms by name and cannot half-specify one.
#
#   bash scripts/run_arms.sh A0 C1 S          # any subset, one GPU each
#   GPUS="4,5,6,7" bash scripts/run_arms.sh   # all six arms
#
# A0 vs A1 separates cosine conditioning from no conditioning; C0 vs C1 separates
# learned conditioning from the combination.  A0 is also run parameter-matched
# (A0m), because plain `agnostic` drops the query cross-attention block and so is
# not a like-for-like control on parameter count.
set -euo pipefail
cd "$(dirname "$0")/.."

# Defaults reproduce /data02/quro/runs/d1_C_full/config.json exactly, so the arm
# is the only thing that changes and C1 doubles as a reproducibility check
# against the historical 26.75 EM.
STEPS="${STEPS:-3000}"
BUDGET="${BUDGET:-8}"
# FIXED_BUDGET=1 trains and evaluates at a single B, with budget dropout off.
#
# Required for the B sweep, not merely tidier.  With mixed budgets the slots are
# generated at the batch maximum and the smaller ones are masked afterwards --
# but slot self-attention carries no budget mask, so a B=8 row batched with B=32
# rows does not produce the same output it would alone.  Sweeping B under mixed
# batches would measure that contamination rather than the budget
# (warning_and_target.md W5, which asks for fixed or budget-grouped batches
# before any sweep).
FIXED_BUDGET="${FIXED_BUDGET:-}"
if [ -n "$FIXED_BUDGET" ]; then
  BUCKETS="$BUDGET"; EVAL_BUDGETS="$BUDGET"; BUDGET_FLAGS=(--no_budget_dropout)
else
  BUCKETS="${BUCKETS:-4,8}"; EVAL_BUDGETS="${EVAL_BUDGETS:-8}"; BUDGET_FLAGS=()
fi
QDROP="${QDROP:-1.0}"
EVAL_MODES="${EVAL_MODES:-D0,D1}"
SEED="${SEED:-42}"
PRESET="${PRESET:-pisco_gonogo}"
# Only the development split by default.  HotpotQA's test half exists to be left
# alone until a protocol is locked (warning_and_target §5.9).
EVAL_FILES="${EVAL_FILES:-trivia=/data02/quro/data/trivia/queries.jsonl}"
RUNS="${QURO_RUNS_DIR:-/data02/quro/runs}"
PREFIX="${PREFIX:-arms}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2,3,4,5,6,7}"

ARMS=("$@")
if [ "${#ARMS[@]}" -eq 0 ]; then
  ARMS=(A0 A0m A1 C0 C1 S P)
fi

COMMIT="$(git rev-parse HEAD)"
DIRTY=""
if ! git diff --quiet HEAD 2>/dev/null; then DIRTY="-dirty"; fi

mkdir -p "$RUNS"

i=0
for arm in "${ARMS[@]}"; do
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  tag="${PREFIX}_${arm}"
  out="$RUNS/$tag"
  mkdir -p "$out"

  # §7 run archiving: the commit, and the working-tree diff when there is one, so
  # a result can still be traced when the tree was not clean at launch.
  echo "${COMMIT}${DIRTY}" > "$out/commit.txt"
  if [ -n "$DIRTY" ]; then git diff HEAD > "$out/worktree.diff"; fi

  # A0m is A0 with the query cross-attention block retained.
  extra=(--arm "${arm%m}")
  if [ "$arm" != "${arm%m}" ]; then extra+=(--agnostic_param_matched); fi

  echo "[gpu $gpu] $tag  (${extra[*]})"
  # --doc_control gives the no-evidence floor that every accuracy number has to be
  # read against.  --query_control is left off here: it now adds two more 2000-row
  # generation passes per arm, and the mismatch diagnostic is only worth paying
  # for once the arm matrix says which arms are worth diagnosing.
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m src.train \
      --preset "$PRESET" --steps "$STEPS" --budget "$BUDGET" \
      --budget_buckets "$BUCKETS" --eval_budgets "$EVAL_BUDGETS" \
      --eval_input_modes "$EVAL_MODES" "${BUDGET_FLAGS[@]}" \
      --query_text_dropout "$QDROP" --seed "$SEED" \
      --num_workers 4 --doc_control \
      --eval_max_samples 2000 \
      --eval_files "$EVAL_FILES" \
      "${extra[@]}" --tag "$tag" --out_dir "$out" \
      > "$RUNS/$tag.log" 2>&1 &
  i=$((i + 1))
done

wait
echo "arms done: ${ARMS[*]}"
# The run record carries arm_label(cfg), i.e. the arm each run actually
# implements.  Check it against the tag before reading any number.
python - "$RUNS" "${ARMS[@]}" <<'EOF'
import json, os, sys
runs, arms = sys.argv[1], sys.argv[2:]
prefix = os.environ.get("PREFIX", "arms")
print(f"{'tag':<16}{'requested':<12}{'implemented':<12}")
for arm in arms:
    path = os.path.join(runs, f"{prefix}_{arm}", "result.json")
    got = "MISSING"
    if os.path.exists(path):
        got = json.load(open(path)).get("arm", "unrecorded")
    flag = "" if got == arm.rstrip("m") or got == arm else "   <-- MISMATCH"
    print(f"{prefix+'_'+arm:<16}{arm:<12}{got:<12}{flag}")
EOF
