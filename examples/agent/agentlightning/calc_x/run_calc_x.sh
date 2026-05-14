#! /bin/bash
set -x

tabs 4
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# calc_x -> agentlightning -> agent -> examples -> repo root (rlinf)
REPO_PATH=$(dirname $(dirname $(dirname $(dirname "$CONFIG_PATH"))))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${REPO_PATH}/examples:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-1.5b-enginehttp-multiturn"
else
    CONFIG_NAME=$1
    shift
fi
python ${CONFIG_PATH}/main.py --config-path ${CONFIG_PATH}/config/ --config-name $CONFIG_NAME "$@"
