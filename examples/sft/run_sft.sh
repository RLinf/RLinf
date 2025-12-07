#! /bin/bash

export SFT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$SFT_PATH"))
export SRC_FILE="${SFT_PATH}/train_sft_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
export LIBERO_REPO_PATH="/opt/libero"

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="pi0_maniskill"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_sft.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${SFT_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
