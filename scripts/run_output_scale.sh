#!/usr/bin/env bash
# Stage C of docs/LATENT_CONTEXTUALISATION.md §3.3: does writing the soft tokens
# at a scale the decoder's layers can actually move change anything?
#
# Three arms, identical except for the scale, so a difference is attributable to
# it and to nothing else.  All three use `--arm P` -- pisco_direct, a readout
# with *zero* parameters -- so the only thing training touches is the decoder
# LoRA and any gain cannot be laundered through a learned readout.
#
#   P1    scale 1.0   PISCO's own scale.  The matched control, and it has to be
#                     run rather than taken from history: the published P numbers
#                     are mostly frozen-decoder, and comparing those to an arm
#                     trained for 3000 steps would credit the scale for the
#                     training.
#   Pa10  scale 0.10  relative per-layer update 0.204 (2.9x PISCO's 0.071),
#                     zero-shot QA deficit -0.055 [-0.115, +0.005]
#   Pa05  scale 0.05  relative update 0.273, the closest any scale gets to the
#                     pre-registered 0.3 criterion; zero-shot deficit -0.135
#
# Two scales rather than one on purpose.  The pre-registered rule ("largest alpha
# with relative update >= 0.3") selected nothing -- no scale reaches 0.3 before
# the document collapses -- so the choice fell back to judgement, and running
# both removes it as a hidden degree of freedom.
#
#   bash scripts/run_output_scale.sh              # all three
#   GPUS="0,1,2" STEPS=3000 bash scripts/run_output_scale.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${STEPS:-3000}"
SEED="${SEED:-42}"
PRESET="${PRESET:-pisco_hotpot}"
# D0 with the real question, which is the setting the 0.520-vs-0.625 gap was
# measured in.  The historical arm runs trained under D1 (query_text_dropout 1.0)
# because the project's target was D1 parity; that is a different question.
QDROP="${QDROP:-0.0}"
EVAL_MODES="${EVAL_MODES:-D0}"
# Dev only.  HotpotQA's test half stays untouched until a protocol is locked.
EVAL_FILES="${EVAL_FILES:-dev=/data02/quro/data/hotpot/dev.jsonl}"
RUNS="${QURO_RUNS_DIR:-/data02/quro/runs}"
PREFIX="${PREFIX:-oscale}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS:-0,1,2}"

COMMIT="$(git rev-parse HEAD)"
DIRTY=""
if ! git diff --quiet HEAD 2>/dev/null; then DIRTY="-dirty"; fi
mkdir -p "$RUNS"

# name:scale
ARMS=("${@:-}")
if [ -z "${ARMS[0]:-}" ]; then ARMS=(P1:1.0 Pa10:0.10 Pa05:0.05); fi

i=0
for spec in "${ARMS[@]}"; do
  name="${spec%%:*}"; scale="${spec##*:}"
  gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
  tag="${PREFIX}_${name}"
  out="$RUNS/$tag"
  mkdir -p "$out"
  echo "${COMMIT}${DIRTY}" > "$out/commit.txt"
  if [ -n "$DIRTY" ]; then git diff HEAD > "$out/worktree.diff"; fi

  echo "[gpu $gpu] $tag  output_scale=$scale"
  CUDA_VISIBLE_DEVICES="$gpu" nohup python -m src.train \
      --preset "$PRESET" --steps "$STEPS" --seed "$SEED" \
      --arm P --output_scale "$scale" --generator_lora_init pisco \
      --budget 8 --budget_buckets 8 --no_budget_dropout --eval_budgets 8 \
      --eval_input_modes "$EVAL_MODES" --query_text_dropout "$QDROP" \
      --eval_files "$EVAL_FILES" --eval_max_samples 2000 \
      --num_workers 4 --doc_control \
      --tag "$tag" --out_dir "$out" \
      > "$RUNS/$tag.log" 2>&1 &
  i=$((i + 1))
done

wait
echo "output-scale arms done"
python - "$RUNS" "$PREFIX" "${ARMS[@]}" <<'EOF'
import json, os, sys
runs, prefix, specs = sys.argv[1], sys.argv[2], sys.argv[3:]
print(f"{'tag':<14}{'scale':>7}{'arm':>6}{'substring':>11}{'em':>8}{'f1':>8}")
for spec in specs:
    name = spec.split(":")[0]
    path = os.path.join(runs, f"{prefix}_{name}", "result.json")
    if not os.path.exists(path):
        print(f"{name:<14}{'':>7}{'MISSING':>6}")
        continue
    r = json.load(open(path))
    scale = r.get("config", {}).get("readout", {}).get("output_scale")
    row = next(iter(r.get("eval", {}).values()), {})
    print(f"{name:<14}{scale!s:>7}{r.get('arm', '?'):>6}"
          f"{row.get('substring', float('nan')):>11.4f}"
          f"{row.get('em', float('nan')):>8.4f}{row.get('f1', float('nan')):>8.4f}")
EOF
