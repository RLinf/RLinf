#!/bin/bash
# Launch script for LLM SFT (text-only, e.g. Qwen3-4B).
export SFT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$SFT_PATH"))
export SRC_FILE="${SFT_PATH}/train_llm_sft.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ -z "$1" ]; then
    CONFIG_NAME="qwen3_sft_llm"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/llm_sft/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_llm_sft.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${SFT_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
