#!/usr/bin/env bash
# Wait for the HotpotQA wave-1 arms to finish, then start wave 2 immediately.
#
# The point is that the GPUs never idle between waves.  Analysis needs a human
# (or a model) in the loop; launching the next four runs does not, so it should
# not wait for one.
#
# Wave 2 completes what wave 1 left out:
#
#   P  (qdrop=1.0)  the upstream method itself, at its own working point --
#                   10 docs x m=8 = 80 soft tokens against C/S's 8.  This is the
#                   only arm that can measure what the second compression costs;
#                   S cannot, because S is also budget 8 and would collapse
#                   alongside C without revealing whether the budget or the
#                   method was at fault.  Multi-hop is the first task where that
#                   question bites, since two gold paragraphs have to survive
#                   the squeeze instead of one.
#   A1 (qdrop=1.0)  the fourth cell of the 2x2, so HotpotQA gets the same
#                   cosine-vs-learned decomposition TriviaQA got.
#   C1, S (qdrop=0) wave 1 trains with the question always removed, so its D0 is
#                   out of distribution and cannot be compared against the
#                   historical D0 numbers.  These two are the in-distribution D0
#                   comparison, which is the main-table setting.
set -euo pipefail
cd "$(dirname "$0")/.."

RUNS="${QURO_RUNS_DIR:-/data02/quro/runs}"
running () { ps -u "$(id -un)" -o cmd= | grep -c '[s]rc\.train --preset' || true; }

echo "[chain] waiting for wave 1 ($(running) training processes alive)"
while [ "$(running)" -gt 0 ]; do sleep 30; done
echo "[chain] wave 1 finished at $(date '+%F %T')"

echo "[chain] ---- wave 1 results ----"
for a in A0 C0 C1 S; do
  echo "--- hp1_$a ---"
  grep "\[eval\]" "$RUNS/hp1_$a.log" 2>/dev/null || echo "(no eval lines -- run failed?)"
done
echo "[chain] ------------------------"

HOTPOT_EVAL="dev=/data02/quro/data/hotpot/dev.jsonl"

# D1-trained pair, same config as wave 1 so the numbers join that table.
PRESET=pisco_hotpot EVAL_FILES="$HOTPOT_EVAL" QDROP=1.0 \
  GPUS="0,1" PREFIX="hp2" \
  nohup bash scripts/run_arms.sh P A1 > /tmp/hp2_launch.log 2>&1 &

# D0-trained pair: the in-distribution main-table setting.
PRESET=pisco_hotpot EVAL_FILES="$HOTPOT_EVAL" QDROP=0.0 \
  GPUS="2,3" PREFIX="hp2d0" \
  nohup bash scripts/run_arms.sh C1 S > /tmp/hp2d0_launch.log 2>&1 &

sleep 90
echo "[chain] wave 2 launched:"
for tag in hp2_P hp2_A1 hp2d0_C1 hp2d0_S; do
  printf "  %-12s " "$tag"
  grep -E "^\[cfg\]" "$RUNS/$tag.log" 2>/dev/null | head -1 || echo "(booting)"
done
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "[chain] done -- wave 2 is running, wave 1 is ready to analyse"
