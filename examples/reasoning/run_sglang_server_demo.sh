#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="sglang_server_demo"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/reasoning/sglang_server_demo.py --config-path ${CONFIG_PATH}/config/  --config-name $CONFIG_NAME
