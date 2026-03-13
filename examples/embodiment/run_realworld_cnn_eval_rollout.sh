#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_realworld_cnn_rollout.py"

if [ -z "${1:-}" ]; then
    CONFIG_NAME="realworld_peginsertion_eval_cnn_rollout"
else
    CONFIG_NAME=$1
fi

CKPT_PATH=${2:-null}
EVAL_EPISODES=${3:-50}
# OUTPUT_DIR=${4:-"${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-eval-${CONFIG_NAME}"}
OUTPUT_DIR=${4:-"/tmp/rlinf_eval_rollouts/${CONFIG_NAME}"}
ENV_SAVE_DIR=${5:-"/tmp/rlinf_eval_rollouts/${CONFIG_NAME}"}

echo "Using Python at $(which python)"
echo "Config: ${CONFIG_NAME}"
echo "CKPT_PATH: ${CKPT_PATH}"
echo "EVAL_EPISODES: ${EVAL_EPISODES}"
echo "OUTPUT_DIR: ${OUTPUT_DIR}"

LOG_DIR="${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/run_eval_rollout.log"
mkdir -p "${LOG_DIR}"

CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${OUTPUT_DIR}"
    "runner.eval_data.save_dir=${ENV_SAVE_DIR}"
    "env.eval.data_collection.save_dir=${ENV_SAVE_DIR}"
    "env.eval.data_collection.export_format=pt"
    "algorithm.eval_rollout_epoch=${EVAL_EPISODES}"
    "runner.ckpt_path=${CKPT_PATH}"
)

printf '%q ' "${CMD[@]}" > "${LOG_FILE}"
echo >> "${LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"