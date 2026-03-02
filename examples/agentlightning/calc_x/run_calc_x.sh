#! /bin/bash
set -x

tabs 4
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname $(dirname "$CONFIG_PATH")))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${REPO_PATH}/examples:$PYTHONPATH

# Check if first argument is eval parameter (starts with "eval=")
if [[ "$1" == eval=* ]] || [[ "$1" == +eval=* ]]; then
    CONFIG_NAME="qwen2.5-1.5b-trajectory"
    # HF eval mode: force eval=true and disable resume_dir auto loading.
    ARGS=()
    for arg in "$@"; do
        if [[ "$arg" == eval=* ]] && [[ "$arg" != +eval=* ]]; then
            ARGS+=("+eval=true")
        else
            ARGS+=("$arg")
        fi
    done
    ARGS+=("+runner.resume_dir=null")
    python ${CONFIG_PATH}/main.py --config-path ${CONFIG_PATH}/config/ --config-name $CONFIG_NAME "${ARGS[@]}"
else
    if [ -z "$1" ]; then
        CONFIG_NAME="qwen2.5-1.5b-trajectory"
    else
        CONFIG_NAME=$1
        shift
    fi
    python ${CONFIG_PATH}/main.py --config-path ${CONFIG_PATH}/config/ --config-name $CONFIG_NAME "$@"
fi

