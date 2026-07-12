#!/usr/bin/env bash
set -euo pipefail

: "${DUALVIEW_SFT_DATA_ROOT:?Set the processed success dataset root}"
: "${QWEN_MODEL_PATH:?Set the Qwen3-VL-4B-Instruct path}"
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PWD}/logs/qwentrend_success_sft"}

python examples/sft/train_vlm_sft.py \
  --config-path "${PWD}/examples/sft/config" \
  --config-name qwen3vl_sft_qwentrend_success \
  actor.model.model_path="${QWEN_MODEL_PATH}" \
  runner.output_dir="${OUTPUT_ROOT}" \
  "$@"
