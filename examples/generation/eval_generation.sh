#!/bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${REPO_PATH}/examples/embodiment/eval_embodied_agent.py"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    CONFIG_NAME=${CONFIG_NAME:-"wan22_ti2v_5b_nft_video_ocr"}
    EXTRA_ARGS=("${@}")
else
    CONFIG_NAME=$1
    EXTRA_ARGS=("${@:2}")
fi

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/eval_generation.log"
mkdir -p "${LOG_DIR}"
CMD=(python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/" --config-name "${CONFIG_NAME}" "runner.logger.log_path=${LOG_DIR}" "${EXTRA_ARGS[@]}")
printf "%q " "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf "\n" >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
