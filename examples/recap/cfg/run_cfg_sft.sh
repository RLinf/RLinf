#!/bin/bash

# CFG SFT Training Launch Script
# Usage: bash examples/recap/cfg/run_cfg_sft.sh [CONFIG_NAME] [ADDITIONAL_ARGS]
# Example: bash examples/recap/cfg/run_cfg_sft.sh libero_cfg_openpi runner.max_epochs=10

export SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export EMBODIED_PATH="${SCRIPT_DIR}"
export SRC_FILE="${SCRIPT_DIR}/train_cfg.py"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Suppress libdav1d/ffmpeg verbose logging
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export OPENCV_LOG_LEVEL=off

export PYTHONPATH="${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH"

source switch_env openpi 2>/dev/null || true

if [ -z "$1" ]; then
    CONFIG_NAME="libero_cfg_openpi"
else
    CONFIG_NAME=$1
    shift
fi

echo "Using Python at $(which python)"
echo "Config: ${CONFIG_NAME}"

LOG_DIR="${REPO_PATH}/logs/cfg_sft/${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_cfg_sft.log"
mkdir -p "${LOG_DIR}"

CMD="python ${SRC_FILE} --config-path ${SCRIPT_DIR}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} $@"
echo "${CMD}" > "${MEGA_LOG_FILE}"
${CMD} 2>&1 | grep --line-buffered -v "libdav1d" | tee -a "${MEGA_LOG_FILE}"
