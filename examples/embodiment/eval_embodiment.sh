#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export LIBERO_PRO_PATH=${LIBERO_PRO_PATH:-"/path/to/LIBERO-PRO"}
export LIBERO_PLUS_PATH=${LIBERO_PLUS_PATH:-"/path/to/liberoplus"}

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export HYDRA_FULL_ERROR=1

# $1: Config Name, $2: Robot Platform, $3: Libero Variant (base, pro, plus)
CONFIG_NAME=${1:-"maniskill_ppo_openvlaoft"}
ROBOT_PLATFORM=${2:-"LIBERO"}
L_TYPE=${3:-"base"}

export ROBOT_PLATFORM
export LIBERO_TYPE=$L_TYPE

if [ "$LIBERO_TYPE" == "pro" ]; then
    export PYTHONPATH="${LIBERO_PRO_PATH}:${PYTHONPATH}"
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export PYTHONPATH="${LIBERO_PLUS_PATH}:${PYTHONPATH}"
    export LIBERO_TYPE="plus"
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Base LIBERO"
fi

export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

export OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-$OMNIGIBSON_DATA_PATH/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

LOG_DIR="${REPO_PATH}/logs/eval/$(date +'%Y%m%d-%H:%M')-${CONFIG_NAME}-${LIBERO_TYPE}"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"

echo "Executing: $CMD"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}