#! /bin/bash
# clear
export VLM_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$VLM_PATH"))
export SRC_FILE="${VLM_PATH}/train_vlm_sft.py"

# Set the Megatron-Mbrdige and Megatron-LM Path
export PYTHONPATH=/path/to/Megatron-Bridge/src:$PYTHONPATH
export PYTHONPATH=/path/to/Megatron-LM:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2_5_sft_vlm"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')" d
MEGA_LOG_FILE="${LOG_DIR}/run_vlm_sft.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${VLM_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}