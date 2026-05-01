#!/bin/bash

# Run Value Model SFT training
# Usage: bash examples/recap/value/run_value_sft.sh [CONFIG_NAME] [EXTRA_ARGS...]
# Example: bash examples/recap/value/run_value_sft.sh libero_sft_value
# Example: bash examples/recap/value/run_value_sft.sh libero_sft_value data.tag=my_tag
# Example: bash examples/recap/value/run_value_sft.sh libero_sft_value data.eval_data_paths="[{dataset_path: /path}]"

export SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
export EMBODIED_PATH="${SCRIPT_DIR}"
export SRC_FILE="${SCRIPT_DIR}/train_value.py"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Suppress libdav1d/ffmpeg verbose logging
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export OPENCV_LOG_LEVEL=off
export FFREPORT=""

export PYTHONPATH="${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH"

source switch_env openpi 2>/dev/null || echo "Warning: switch_env not found, using current environment"

if [ -z "$1" ]; then
    CONFIG_NAME="libero_sft_value"
else
    CONFIG_NAME=$1
fi
shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/value_sft/${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_value_sft.log"
mkdir -p "${LOG_DIR}"
HYDRA_ARGS=("runner.logger.log_path=${LOG_DIR}")
CMD_BASE="python ${SRC_FILE} --config-path ${SCRIPT_DIR}/config/ --config-name ${CONFIG_NAME}"
echo "${CMD_BASE} ${HYDRA_ARGS[*]} ${EXTRA_ARGS}" > "${MEGA_LOG_FILE}"
${CMD_BASE} "${HYDRA_ARGS[@]}" ${EXTRA_ARGS} 2>&1 | grep -v "libdav1d" | tee -a "${MEGA_LOG_FILE}"
