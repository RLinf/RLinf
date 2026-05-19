#! /bin/bash
set -x

unset https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

CONFIG_NAME=${1:-qwen2.5-1.5b-enginehttp-multiturn}

python ${REPO_PATH}/examples/agent/agentlightning/calc_x/main.py --config-path ${REPO_PATH}/tests/e2e_tests/agentlightning --config-name ${CONFIG_NAME}
