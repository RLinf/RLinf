#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

export DREAMZERO_PATH=${DREAMZERO_PATH:-"/path/to/DreamZero"}
export PYTHONPATH=${DREAMZERO_PATH}:$PYTHONPATH

export LINGBOT_VA_REPO_PATH=${LINGBOT_VA_REPO_PATH:-"/path/to/lingbot-va"}
export LINGBOT_VA_MODEL_PATH=${LINGBOT_VA_MODEL_PATH:-"/path/to/lingbot-va-base"}
export LINGBOT_VA_DATASET_PATH=${LINGBOT_VA_DATASET_PATH:-"/path/to/libero-data"}
export PYTHONPATH=${LINGBOT_VA_REPO_PATH}:$PYTHONPATH

# Required by the 5B-param transformer to fit alongside activations on a
# single GPU; without this the allocator fragments and OOMs during model
# wrap or first forward.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-"expandable_segments:True"}

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}