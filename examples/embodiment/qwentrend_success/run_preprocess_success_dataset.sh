#!/usr/bin/env bash
set -euo pipefail

: "${UNIFORM_DATA_ROOT:?Set the root containing step0, step20, ..., step200}"
: "${DUALVIEW_SFT_DATA_ROOT:?Set the processed dataset output root}"

raw_args=()
for step in 0 20 40 60 80 100 120 140 160 180 200; do
  raw_args+=(--raw-data-path "${UNIFORM_DATA_ROOT}/step${step}")
done

python examples/reward/preprocess_qwentrend_terminal_success_dataset.py \
  "${raw_args[@]}" \
  --output-dir "${DUALVIEW_SFT_DATA_ROOT}" \
  --window-size 5 \
  --val-split 0.1 \
  --max-positive 8000 \
  --negative-positive-ratio 3 \
  --hard-negatives-per-episode 3 \
  --success-exclusion-steps 8 \
  --workers 32 \
  --seed 42
