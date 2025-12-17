#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/collect_real_data.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
export LIBERO_REPO_PATH="/opt/libero"

export GLOO_SOCKET_IFNAME=eno1 # 

export ROBOT_PYTHONPATH=/home/franka/catkin_franka/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:/home/franka/RLinf
export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:${ROBOT_PYTHONPATH}:$PYTHONPATH

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1


if [ -z "$1" ]; then
    CONFIG_NAME="real_collect_data"
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