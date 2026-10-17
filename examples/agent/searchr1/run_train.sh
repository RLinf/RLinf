#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname $(dirname "$CONFIG_PATH")))
if [ -z "${MEGATRON_PATH:-}" ]; then
    if [ -d "${VIRTUAL_ENV:-/opt/venv/reason}/Megatron-LM" ]; then
        MEGATRON_PATH="${VIRTUAL_ENV:-/opt/venv/reason}/Megatron-LM"
    else
        MEGATRON_PATH=/opt/Megatron-LM
    fi
fi
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${REPO_PATH}/examples:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="train_qwen2.5"
else
    CONFIG_NAME=$1
    shift
fi

exec python ${REPO_PATH}/examples/agent/searchr1/train.py --config-path ${CONFIG_PATH}/config/  --config-name "$CONFIG_NAME" "$@"
