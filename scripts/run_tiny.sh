#!/usr/bin/env bash
# Local prototype smoke test. It validates contracts, not research quality.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/make_demo_data.py --out_dir data/multiq
python3 tests/test_shapes.py
python3 -m src.train --preset tiny --stage stage1 --steps "${STEPS:-20}" \
  --eval_files "seen=data/multiq/eval_seen.jsonl" \
  --output_query_mode xattn --query_control --out_dir runs/tiny_v0
