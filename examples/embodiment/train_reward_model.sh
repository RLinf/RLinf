#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/main_reward.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Optional: set CUDA devices
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Optional: set MuJoCo rendering backend (if needed)
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# Get config name from command line argument, default to train_reward_model
if [ -z "$1" ]; then
    CONFIG_NAME="train_reward_model"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/reward_training/$(date +'%Y%m%d-%H%M%S')"
MEGA_LOG_FILE="${LOG_DIR}/train_reward_model.log"
mkdir -p "${LOG_DIR}"

# Build command with config path and log path override
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"

# Append any additional arguments passed to the script
shift  # Remove first argument (config name)
if [ $# -gt 0 ]; then
    CMD="${CMD} $@"
fi

echo "Command: ${CMD}" | tee ${MEGA_LOG_FILE}
echo "Log directory: ${LOG_DIR}" | tee -a ${MEGA_LOG_FILE}
echo "Starting training..." | tee -a ${MEGA_LOG_FILE}
echo "----------------------------------------" | tee -a ${MEGA_LOG_FILE}

${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
