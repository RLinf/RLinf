#!/bin/bash

# CFG SFT Training Launch Script
# Usage: bash run_cfg_sft.sh [CONFIG_NAME] [ADDITIONAL_ARGS]
# Example: bash run_cfg_sft.sh libero_cfg_sft runner.max_epochs=10

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_cfg_sft.py"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Suppress libdav1d/ffmpeg verbose logging
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet

export PYTHONPATH=${REPO_PATH}:${VLA_LIB_PATH:+${VLA_LIB_PATH}:}${LIBERO_REPO_PATH}:$PYTHONPATH

# Activate the openpi environment
source switch_env openpi 2>/dev/null || true

if [ -z "$1" ]; then
    CONFIG_NAME="libero_cfg_sft"
else
    CONFIG_NAME=$1
    shift  # Remove first argument so $@ contains only additional args
fi

echo "Using Python at $(which python)"
echo "Config: ${CONFIG_NAME}"

LOG_DIR="${REPO_PATH}/logs/cfg_sft/${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_cfg_sft.log"
mkdir -p "${LOG_DIR}"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} $@"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
