#! /bin/bash

# Run vla_lib Value Model SFT training
# Usage: bash run_vla_lib_sft.sh [CONFIG_NAME] [EVAL_DATASET_PATH]
# Example: bash run_vla_lib_sft.sh libero_value_sft /path/to/eval_dataset

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_vla_lib_sft.py"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Suppress libdav1d/ffmpeg verbose logging
export AV_LOG_FORCE_NOCOLOR=1
export LIBAV_LOG_LEVEL=quiet
export FFREPORT=""

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

# Activate the openpi environment
source switch_env openpi 2>/dev/null || echo "Warning: switch_env not found, using current environment"

if [ -z "$1" ]; then
    CONFIG_NAME="libero_value_sft"
else
    CONFIG_NAME=$1
fi

if [ -z "$2" ]; then
    EVAL_DATASET_PATH=""
else
    EVAL_DATASET_PATH=$2
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/vla_lib_sft/${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_vla_lib_sft.log"
mkdir -p "${LOG_DIR}"
HYDRA_OVERRIDES="runner.logger.log_path=${LOG_DIR}"
if [ -n "${EVAL_DATASET_PATH}" ]; then
    HYDRA_OVERRIDES="${HYDRA_OVERRIDES} data.eval_dataset_path=${EVAL_DATASET_PATH}"
fi
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} ${HYDRA_OVERRIDES}"
echo ${CMD} > ${MEGA_LOG_FILE}
# Filter out libdav1d verbose logging
${CMD} 2>&1 | grep -v "libdav1d" | tee -a ${MEGA_LOG_FILE}
