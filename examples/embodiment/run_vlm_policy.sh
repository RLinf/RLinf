#!/usr/bin/env bash
# Launch VLM-as-Policy. Default config: libero_gso_grpo_pear_sglang.
# Usage: bash examples/embodiment/run_vlm_policy.sh [config_name] [hydra overrides...]
set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"

export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

CONFIG_NAME="${CONFIG_NAME:-libero_gso_grpo_pear_sglang}"
if [[ $# -gt 0 && "$1" != *=* ]]; then
  CONFIG_NAME="$1"
  shift
fi

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
mkdir -p "${LOG_DIR}"
cd "${REPO_PATH}"

echo "Logging to ${LOG_DIR}/run_embodiment.log"
python "${EMBODIED_PATH}/train_embodied_agent.py" \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name "${CONFIG_NAME}" \
  "runner.logger.log_path=${LOG_DIR}" \
  "$@" 2>&1 | tee -a "${LOG_DIR}/run_embodiment.log"
