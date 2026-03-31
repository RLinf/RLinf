#! /bin/bash

export EMBODIED_PATH="/mnt/public/yuanhua/RLinf/examples/embodiment"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export STEPS=${STEPS:-3}

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

# Base path to the BEHAVIOR dataset, which is the BEHAVIOR-1k repo's dataset folder
# Only required when running the behavior experiment.
export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
# Base path to Isaac Sim, only required when running the behavior experiment.
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"MANISKILL"}}

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/placement_test_logs/${CONFIG_NAME}-$(date +'%m-%d-%H-%M')" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path /mnt/public/yuanhua/RLinf/placement_test --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} \
    runner.max_epochs=${STEPS}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}