#! /bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
# NOTE: set LIBERO_REPO_PATH to the path of the LIBERO repo
export LIBERO_REPO_PATH="/path/to/repo/LIBERO"
# NOTE: set LIBERO_CONFIG_PATH for libero/libero/__init__.py
export LIBERO_CONFIG_PATH=${LIBERO_REPO_PATH}

export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1


if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvla_eval"
else
    CONFIG_NAME=$1
fi

for env_id in \
    "PutOnPlateInScene25VisionImage-v1" "PutOnPlateInScene25VisionTexture03-v1" "PutOnPlateInScene25VisionTexture05-v1" \
    "PutOnPlateInScene25VisionWhole03-v1"  "PutOnPlateInScene25VisionWhole05-v1" \
    "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25Plate-v1" "PutOnPlateInScene25Instruct-v1" \
    "PutOnPlateInScene25MultiCarrot-v1" "PutOnPlateInScene25MultiPlate-v1" \
    "PutOnPlateInScene25Position-v1" "PutOnPlateInScene25EEPose-v1" "PutOnPlateInScene25PositionChangeTo-v1" ; \
do
    obj_set="test"
    LOG_DIR="${REPO_PATH}/logs/eval/$(date +'%Y%m%d-%H:%M:%S')-${env_id}-${obj_set}"
    MEGA_LOG_FILE="${LOG_DIR}/run_ppo.log"
    mkdir -p "${LOG_DIR}"
    CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} \
        runner.logger.log_path=${LOG_DIR} \
        env.eval.init_params.id=${env_id} \
        env.eval.init_params.obj_set=$obj_set"

    echo ${CMD} > ${MEGA_LOG_FILE}
    ${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
done