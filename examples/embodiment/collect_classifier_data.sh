#!/bin/bash
# Collect success/failure classifier data via teleoperation.
#
# Usage:
#   bash examples/embodiment/collect_classifier_data.sh [config_name] [env_name]
#
# Examples:
#   bash examples/embodiment/collect_classifier_data.sh                          # defaults
#   bash examples/embodiment/collect_classifier_data.sh realworld_collect_data dex_pnp

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/collect_classifier_data.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    CONFIG_NAME="realworld_collect_data"
else
    CONFIG_NAME=$1
fi

ENV_NAME="${2:-dex_pnp}"

# Ensure DISPLAY is forwarded to Ray workers for camera preview window
if [ -n "$DISPLAY" ]; then
    export RAY_RUNTIME_ENV_DISPLAY="$DISPLAY"
fi

# Avoid X11 shared-memory errors in Docker containers with small /dev/shm.
export QT_X11_NO_MITSHM=1

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-reward-classifier-${ENV_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} env.eval.ignore_terminations=True"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
