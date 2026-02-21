#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_debug_one_iter.py"

export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"

# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
export LIBERO_REPO_PATH="/opt/libero"

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"

# Suppress libdav1d/PyAV/FFmpeg verbose logging (inherited by Python and Ray workers)
export AV_LOG_LEVEL=-8
export LIBAV_LOG_LEVEL=quiet

if [ -z "$1" ]; then
    CONFIG_NAME="libero_10_pi06_reset_base_new_load_buffer_guide10_capacity2048"
else
    CONFIG_NAME=$1
fi

echo "========================================"
echo "Debug Pi06 Training"
echo "========================================"
echo "Config: $CONFIG_NAME"
echo "Python: $(which python)"
echo "========================================"

LOG_DIR="${REPO_PATH}/logs/debug_${CONFIG_NAME}-$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_debug_pi06.log"
mkdir -p "${LOG_DIR}"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
# Run once; filter libdav1d lines from console and log
${CMD} 2>&1 | grep --line-buffered -v "libdav1d" | tee -a ${MEGA_LOG_FILE}