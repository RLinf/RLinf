#! /bin/bash
# auto profile (if needed) + auto placement for embodied training.
# Requires Ray to be running (e.g. ray start --head).
set -x

tabs 4

CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../examples/embodiment" && pwd)"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export REPO_PATH
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

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


ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"

CONFIG_NAME="${1:-maniskill_ppo_openvlaoft_quickstart}"

python ${REPO_PATH}/toolkits/auto_placement/collect_profile.py \
    --config-path ${EMBODIED_PATH}/config \
    --config-name $CONFIG_NAME \