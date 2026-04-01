#!/bin/bash

# ==============================================================================
# 1. Path and Environment Setup
# ==============================================================================
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# ==============================================================================
# 2. BEHAVIOR & OmniGibson Settings (From Official Script)
# Only required when running the behavior experiment.
# ==============================================================================
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}

# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

# ==============================================================================
# 3. Process Cleanup
# Clean up old python training processes to prevent port conflicts or OOM errors
# ==============================================================================
echo "🧹 Cleaning up old python training processes..."
ps -ef | grep "python ${SRC_FILE}" | grep -v grep | awk '{print $2}' | xargs -r kill -9

# ==============================================================================
# 4. Argument Parsing
# ==============================================================================
if [ -z "$1" ]; then
    CONFIG_NAME="gr00t_16_single_gpu"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization)
# Supported platforms are LIBERO, ALOHA, BRIDGE. Default is LIBERO.
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}
export ROBOT_PLATFORM

echo "🤖 Using CONFIG_NAME=${CONFIG_NAME}"
echo "🤖 Using ROBOT_PLATFORM=${ROBOT_PLATFORM}"
echo "🐍 Using Python at $(which python)"

# ==============================================================================
# 5. Single-GPU Overrides
# Force the framework's placement strategy to align with a single GPU
# ==============================================================================
EXTRA_ARGS="actor.enable_offload=False \
rollout.enable_offload=False \
env.train.total_num_envs=32 \
env.train.group_size=8 \
env.eval.total_num_envs=1 \
env.eval.group_size=1 \
actor.global_batch_size=512 \
actor.micro_batch_size=32 \
algorithm.rollout_epoch=1 \
actor.model.model_type=gr00t_1_6 \
actor.model.action_dim=128"

# ==============================================================================
# 6. Logging and Execution
# ==============================================================================
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"

# Build the final execution command
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} ${EXTRA_ARGS}"

echo "------------------------------------------------"
echo "🔥 Final Command: ${CMD}"
echo "------------------------------------------------"

# Save the command to the log file for reproducibility
echo "${CMD}" > "${MEGA_LOG_FILE}"

# Execute the command and pipe output to both console and log file
${CMD} 2>&1 | tee -a "${MEGA_LOG_FILE}"