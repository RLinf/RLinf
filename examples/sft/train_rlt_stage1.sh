#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_rlt_stage1.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="rlt_stage1_maniskill_joint"
else
    CONFIG_NAME=$1
fi
shift $(( $# > 0 ? 1 : 0 ))

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/train_rlt_stage1.log"
mkdir -p "${LOG_DIR}"

CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    runner.logger.log_path="${LOG_DIR}"
)

if [ "$#" -gt 0 ]; then
    CMD+=("$@")
fi

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
