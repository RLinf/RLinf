#! /bin/bash
export HF_ENDPOINT=https://hf-mirror.com

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

EVAL_NAME=test_roboverse
CKPT_PATH=/mnt/public/xyq/openvla-7b        # .pt file
CONFIG_NAME=roboverse_grpo_openvla      # env.eval must be maniskill_ood_template

LOG_DIR="${REPO_PATH}/logs/eval/${EVAL_NAME}/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_ppo.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ \
    --config-name ${CONFIG_NAME} \
    runner.logger.log_path=${LOG_DIR}"

echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
