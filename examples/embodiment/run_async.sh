#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_async.py"

export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
# setup_sim_env is defined in sim_env.sh.
source "${EMBODIED_PATH}/sim_env.sh"
setup_sim_env

if [ -z "$1" ]; then
    CONFIG_NAME=${CONFIG_NAME:-"maniskill_sac_mlp_async"}
else
    CONFIG_NAME=$1
    shift
fi

# NOTE: Set the active robot platform (required for correct action dimension and normalization), supported platforms are LIBERO, ALOHA, BRIDGE, default is LIBERO
if [ "$#" -gt 0 ] && [[ "$1" != *=* && "$1" != +* && "$1" != ~* && "$1" != -* ]]; then
    ROBOT_PLATFORM="$1"
    shift
else
    ROBOT_PLATFORM=${ROBOT_PLATFORM:-"LIBERO"}
fi

export ROBOT_PLATFORM
echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
    "$@"
)
echo "${CMD[*]}" > "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
