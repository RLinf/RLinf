#! /bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

EVAL_NAME=YOUR_EVAL_NAME
CKPT_PATH=YOUR_CKPT_PATH        # .pt file
CONFIG_NAME=YOUR_CFG_NAME       # env.eval must be maniskill_ood_template

for env_id in \
    "PutOnPlateInScene25VisionImage-v1" "PutOnPlateInScene25VisionTexture03-v1" "PutOnPlateInScene25VisionTexture05-v1" \
    "PutOnPlateInScene25VisionWhole03-v1"  "PutOnPlateInScene25VisionWhole05-v1" \
    "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25Plate-v1" "PutOnPlateInScene25Instruct-v1" \
    "PutOnPlateInScene25MultiCarrot-v1" "PutOnPlateInScene25MultiPlate-v1" \
    "PutOnPlateInScene25Position-v1" "PutOnPlateInScene25EEPose-v1" "PutOnPlateInScene25PositionChangeTo-v1" ; \
do
    obj_set="test"
    LOG_DIR="${REPO_PATH}/logs/eval/${EVAL_NAME}/$(date +'%Y%m%d-%H:%M:%S')-${env_id}-${obj_set}"
    MEGA_LOG_FILE="${LOG_DIR}/run_ppo.log"
    mkdir -p "${LOG_DIR}"
    CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ \
        --config-name ${CONFIG_NAME} \
        runner.logger.log_path=${LOG_DIR} \
        env.eval.init_params.id=${env_id} \
        env.eval.init_params.obj_set=${obj_set} \
        actor.model.ckpt_path=${CKPT_PATH}"

    echo ${CMD} > ${MEGA_LOG_FILE}
    ${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
done

for env_id in \
    "PutOnPlateInScene25Carrot-v1" "PutOnPlateInScene25MultiCarrot-v1" \
    "PutOnPlateInScene25MultiPlate-v1" ; \
do
    obj_set="train"
    LOG_DIR="${REPO_PATH}/logs/eval/${EVAL_NAME}/$(date +'%Y%m%d-%H:%M:%S')-${env_id}-${obj_set}"
    MEGA_LOG_FILE="${LOG_DIR}/run_ppo.log"
    mkdir -p "${LOG_DIR}"
    CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ \
        --config-name ${CONFIG_NAME} \
        runner.logger.log_path=${LOG_DIR} \
        env.eval.init_params.id=${env_id} \
        env.eval.init_params.obj_set=${obj_set} \
        actor.model.ckpt_path=${CKPT_PATH}"
    echo ${CMD}  > ${MEGA_LOG_FILE}
    ${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
done