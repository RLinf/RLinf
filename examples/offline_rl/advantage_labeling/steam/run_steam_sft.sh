#!/bin/bash

# Run STEAM value model SFT training
# Usage: bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh [CONFIG_NAME] [EXTRA_ARGS...]
# Example: bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh steam_value_model_sft

export SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export OFFLINE_RL_CONFIG="${REPO_PATH}/examples/offline_rl/config"
export EMBODIED_PATH="${SCRIPT_DIR}"
export SRC_FILE="${SCRIPT_DIR}/train_steam.py"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export OPENCV_LOG_LEVEL=off
export FFREPORT=""

export PYTHONPATH="${REPO_PATH}:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source switch_env openpi 2>/dev/null || echo "Warning: switch_env not found, using current environment"

if [ -z "$1" ]; then
    CONFIG_NAME="steam_value_model_sft"
else
    CONFIG_NAME=$1
fi
shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/steam_sft/${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
LOG_FILE="${LOG_DIR}/run_steam_sft.log"
mkdir -p "${LOG_DIR}"
HYDRA_ARGS=("runner.logger.log_path=${LOG_DIR}")
CMD_BASE="python ${SRC_FILE} --config-path ${OFFLINE_RL_CONFIG} --config-name ${CONFIG_NAME}"
echo "${CMD_BASE} ${HYDRA_ARGS[*]} ${EXTRA_ARGS}" > "${LOG_FILE}"
${CMD_BASE} "${HYDRA_ARGS[@]}" ${EXTRA_ARGS} 2>&1 | grep -v "libdav1d" | tee -a "${LOG_FILE}"
exit "${PIPESTATUS[0]}"
