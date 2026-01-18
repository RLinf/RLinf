#!/bin/bash
# Train ResNet Reward Model using RLinf framework
#
# Usage:
#   ./train_reward_model.sh [DATA_PATH]
#
# Example:
#   ./train_reward_model.sh /path/to/collected_data

export REWARD_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$REWARD_PATH"))
export SRC_FILE="${REPO_PATH}/examples/reward/train_reward_model.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Default data path (override with first argument)
DATA_PATH="$1"


# Convert to absolute path if relative
if [[ ! "$DATA_PATH" = /* ]]; then
    DATA_PATH="${REPO_PATH}/${DATA_PATH}"
fi

# Verify data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo "ERROR: Data path does not exist: $DATA_PATH"
    exit 1
fi

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-reward_training"
LOG_FILE="${LOG_DIR}/training.log"

mkdir -p "${LOG_DIR}"

echo "============================================"
echo "ResNet Reward Model Training (RLinf)"
echo "============================================"
echo "Data path: $DATA_PATH"
echo "Log dir: $LOG_DIR"
echo "============================================"

DEBUG_SAVE_DIR="${LOG_DIR}/training_data_debug"

CMD="python ${SRC_FILE} \
    runner.logger.log_path=${LOG_DIR} \
    data.data_path=${DATA_PATH} \
    data.debug_save_dir=${DEBUG_SAVE_DIR}"

echo "Running: ${CMD}"
echo "${CMD}" > "${LOG_FILE}"

${CMD} 2>&1 | tee -a "${LOG_FILE}"

