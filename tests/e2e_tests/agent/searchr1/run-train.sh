#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME=qwen2.5-1.5b-megatron
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/agent/searchr1/train.py --config-path ${REPO_PATH}/tests/e2e_tests/agent/searchr1  --config-name $CONFIG_NAME
