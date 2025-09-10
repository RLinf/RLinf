#! /bin/bash
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH=${REPO_PATH}
# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
export PYTHONPATH="/mnt/mnt/public/mjwei/repo/LIBERO":$PYTHONPATH
# NOTE: set LIBERO_CONFIG_PATH for libero/libero/__init__.py
# export LIBERO_CONFIG_PATH=${LIBERO_REPO_PATH}
echo "LIBERO_CONFIG_PATH: ${LIBERO_CONFIG_PATH}"

export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH
echo "PYTHONPATH: ${PYTHONPATH}"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1


if [ -z "$1" ]; then
    CONFIG_NAME="libero_grpo_pi0"
else
    CONFIG_NAME=$1
fi

LOG_DIR="${REPO_PATH}/logs/pi0_4GPU_$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_ppo.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD}
${CMD} 2>&1 | tee ${MEGA_LOG_FILE}
