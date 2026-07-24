#!/usr/bin/env bash
# Step 3a (sparse): LoRA-SFT Qwen3-VL for terminal success 0/1 generation.
set -euo pipefail

: "${DUALVIEW_SFT_DATA_ROOT:?Set the processed sparse success dataset root}"
: "${QWEN_MODEL_PATH:?Set the Qwen3-VL-4B-Instruct path}"
OUTPUT_ROOT=${OUTPUT_ROOT:-"${PWD}/logs/qwentrend_sparse_success_sft"}
PLACEMENT=${PLACEMENT:-0-3}
RAY_DIR=${RAY_DIR:-/dev/shm/qsft_$$}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/openvla/bin/python}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
MAX_STEPS=${MAX_STEPS:-400}
MAX_EPOCHS=${MAX_EPOCHS:-2}
SUCCESS_WEIGHT=${SUCCESS_WEIGHT:-2.22060897}
mkdir -p "${RAY_DIR}/ray" "${RAY_DIR}/tmp"

RAY_TMPDIR="${RAY_DIR}/ray" \
  RLINF_FORCE_LOCAL_RAY=1 \
  RLINF_RAY_TEMP_DIR="${RAY_DIR}/ray" \
  TMPDIR="${RAY_DIR}/tmp" \
  "${PYTHON_BIN}" examples/sft/train_vlm_sft.py \
  --config-path "${PWD}/examples/sft/config" \
  --config-name qwen3vl_sft_qwentrend_success \
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
  actor.model.sample_reweight.enabled=true \
  actor.model.sample_reweight.success_weight="${SUCCESS_WEIGHT}" \
  actor.model.sample_reweight.non_success_weight=1.0 \
  actor.optim.lr=1.0e-5 \
  actor.optim.weight_decay=0.01 \
  actor.optim.lr_scheduler=cosine \
  actor.optim.total_training_steps="${MAX_STEPS}" \
  actor.optim.lr_warmup_steps=20 \
  "$@"
