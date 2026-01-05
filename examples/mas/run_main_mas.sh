#! /bin/bash
set -x

# ray stop
# ray start --head

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export RAY_DEBUG=1

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="mas_train_qwen3_8b_format"
else
    CONFIG_NAME=$1
fi


python ${REPO_PATH}/examples/mas/main_mas.py --config-path ${CONFIG_PATH}/config/  --config-name $CONFIG_NAME 

