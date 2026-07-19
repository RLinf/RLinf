#!/usr/bin/env bash
# Step 3b (dense): LoRA-SFT Qwen3-VL for potential / progress trend labels.
set -euo pipefail

: "${POTENTIAL_SFT_DATA_ROOT:?Set the processed dense potential dataset root}"
: "${QWEN_MODEL_PATH:?Set the Qwen3-VL-4B-Instruct path}"
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PWD}/logs/qwentrend_dense_potential_sft"}
PLACEMENT=${PLACEMENT:-0-3}
RAY_DIR=${RAY_DIR:-/dev/shm/qpft_$$}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/openvla/bin/python}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
MAX_STEPS=${MAX_STEPS:-400}
MAX_EPOCHS=${MAX_EPOCHS:-400}
mkdir -p "${RAY_DIR}/ray" "${RAY_DIR}/tmp" "${OUTPUT_ROOT}"

# qwen3vl_sft_qwentrend.yaml reads DUALVIEW_SFT_DATA_ROOT for train/eval jsonl.
export DUALVIEW_SFT_DATA_ROOT="${POTENTIAL_SFT_DATA_ROOT}"

RAY_TMPDIR="${RAY_DIR}/ray" \
  RLINF_FORCE_LOCAL_RAY=1 \
  RLINF_RAY_TEMP_DIR="${RAY_DIR}/ray" \
  TMPDIR="${RAY_DIR}/tmp" \
  "${PYTHON_BIN}" examples/sft/train_vlm_sft.py \
  --config-path "${PWD}/examples/sft/config" \
  --config-name qwen3vl_sft_qwentrend \
  cluster.component_placement.actor="${PLACEMENT}" \
  actor.model.model_path="${QWEN_MODEL_PATH}" \
  runner.output_dir="${OUTPUT_ROOT}" \
  runner.max_epochs="${MAX_EPOCHS}" \
  runner.max_steps="${MAX_STEPS}" \
  runner.val_check_interval=100 \
  runner.save_interval=100 \
  actor.micro_batch_size="${MICRO_BATCH_SIZE}" \
  actor.eval_batch_size=16 \
  actor.global_batch_size="${GLOBAL_BATCH_SIZE}" \
  actor.optim.lr=1.0e-5 \
  actor.optim.weight_decay=0.01 \
  actor.optim.lr_scheduler=cosine \
  actor.optim.total_training_steps="${MAX_STEPS}" \
  actor.optim.lr_warmup_steps=20 \
  "$@"

# Prefer the final saved step when present; otherwise pick the highest global_step_*.
POT_CKPT=$(find "${OUTPUT_ROOT}" -type d -name "global_step_${MAX_STEPS}" 2>/dev/null | head -1)
if [[ -z "${POT_CKPT}" ]]; then
  POT_CKPT=$(find "${OUTPUT_ROOT}" -type d -name 'global_step_*' 2>/dev/null \
    | sort -t_ -k3 -n | tail -1)
fi
if [[ -n "${POT_CKPT}" ]]; then
  echo "${POT_CKPT}" > "${OUTPUT_ROOT}/SELECTED_CKPT.txt"
  echo "Wrote ${OUTPUT_ROOT}/SELECTED_CKPT.txt -> ${POT_CKPT}"
fi
