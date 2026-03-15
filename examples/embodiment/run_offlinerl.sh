#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
export REPO_PATH=$(dirname "$(dirname "$EMBODIED_PATH")")
export SRC_FILE="${EMBODIED_PATH}/train_offlinerl.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"

if [ -z "$1" ]; then
    CONFIG_NAME="d4rl_iql_mujoco"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_offlinerl.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} runner.logger.experiment_name=${CONFIG_NAME}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
