#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
# Default driver. Overridden below per actor.model.model_type so that
# policies which maintain inter-chunk state (e.g. LingBot-VA's KV-cache
# replay) can use their dedicated single-process driver. See
# eval_lingbotva.py for the reasoning.
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

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

export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

# Select eval driver based on actor.model.model_type in the config YAML.
# Most policies use the generic, multi-worker eval_embodied_agent.py.
# Policies that need inter-chunk model state (KV-cache replay, episode
# buffers) need a dedicated driver that captures per-step raw obs and
# calls their record_chunk_observations / reset_episode hooks. To opt in,
# add a top-level "model_type:" key matching the policy name and provide
# a per-policy driver file (eval_<model_type>.py).
CONFIG_FILE="${EMBODIED_PATH}/config/${CONFIG_NAME}.yaml"
if [ -f "${CONFIG_FILE}" ]; then
    MODEL_TYPE=$(awk -F':' '/^[[:space:]]*model_type:/ {gsub(/[[:space:]"'\'',]/,"",$2); print $2; exit}' "${CONFIG_FILE}")
fi
case "${MODEL_TYPE}" in
    lingbotva)
        CANDIDATE="${EMBODIED_PATH}/eval_lingbotva.py"
        if [ -f "${CANDIDATE}" ]; then
            SRC_FILE="${CANDIDATE}"
        fi
        ;;
esac
echo "Eval driver: ${SRC_FILE} (model_type=${MODEL_TYPE:-unknown})"

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
ROBOT_PLATFORM=${2:-${ROBOT_PLATFORM:-"LIBERO"}}

export ROBOT_PLATFORM

# Libero variant: standard, pro, plus
export LIBERO_TYPE=${LIBERO_TYPE:-"standard"}
if [ "$LIBERO_TYPE" == "pro" ]; then
    export LIBERO_PERTURBATION="all"  # all,swap,object,lan
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" == "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"
# Forward any extra positional args ($3..$N) as Hydra overrides so callers
# can do e.g. `bash eval_embodiment.sh <cfg> <platform> env.eval.task_id_filter=[3]`.
shift $(( $# >= 2 ? 2 : $# ))
echo "python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} $*"
python "${SRC_FILE}" \
    --config-path "${EMBODIED_PATH}/config/" \
    --config-name "${CONFIG_NAME}" \
    runner.logger.log_path="${LOG_DIR}" \
    "$@" 2>&1 | tee "${MEGA_LOG_FILE}"
