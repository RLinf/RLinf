#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${PYTHONPATH}

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-3b-tool-1node"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/mas-search/main_mas_search.py --config-path ${CONFIG_PATH}/config/  --config-name $CONFIG_NAME
