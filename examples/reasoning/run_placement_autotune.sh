#! /bin/bash
set -x

tabs 4

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-1.5b-grpo-megatron"
else
    CONFIG_NAME=$1
fi


python ${REPO_PATH}/toolkits/auto_placement/auto_placement_worker.py \
    --config-path ${CONFIG_PATH}/config/math \
    --config-name $CONFIG_NAME \