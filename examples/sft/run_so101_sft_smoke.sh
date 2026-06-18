#!/usr/bin/env bash
# SO101 pi0 SFT smoke launcher.
#
# Runs a small number of training steps against the merged SO101 dataset to
# prove the full pipeline works.  Override paths via env vars when needed:
#
#   SO101_MODEL_PATH    pi0 base checkpoint dir
#                       (default: /root/pi0_base_so101)
#   SO101_DATA_REPO_ID  LeRobot dataset repo_id (resolved against HF_LEROBOT_HOME)
#                       (default: so101_data)
#   SO101_MAX_STEPS     stop after this many SFT steps (default: 20)
#   SO101_LOG_DIR       where logs/checkpoints land (default: /root/so101_sft_run)
#
# All HF_HUB / network env vars are forced into offline mode so the loader
# never tries to validate the dataset against the (unreachable) HF Hub.
set -euo pipefail

REPO_PATH=/root/RLinf
export EMBODIED_PATH="${REPO_PATH}/examples/sft"
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1

MODEL_PATH="${SO101_MODEL_PATH:-/root/pi0_base_so101}"
DATA_REPO_ID="${SO101_DATA_REPO_ID:-so101_data}"
MAX_STEPS="${SO101_MAX_STEPS:-20}"
LOG_DIR="${SO101_LOG_DIR:-/root/so101_sft_run}"
mkdir -p "${LOG_DIR}"

PY="${REPO_PATH}/.venv/bin/python"
CMD=(
  "${PY}" "${SRC_FILE}"
  --config-path "${EMBODIED_PATH}/config/"
  --config-name so101_sft_openpi
  "runner.logger.log_path=${LOG_DIR}"
  "runner.max_steps=${MAX_STEPS}"
  "actor.model.model_path=${MODEL_PATH}"
  "data.train_data_paths=${DATA_REPO_ID}"
)

echo "[so101-sft-smoke] CMD: ${CMD[*]}"
cd "${REPO_PATH}"
"${CMD[@]}" 2>&1 | tee "${LOG_DIR}/run.log"
