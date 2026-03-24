#!/usr/bin/env bash
set -x

export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Spider SQLite：train 用 database/，eval 用 test_database/；与 yaml 里 parquet 路径无关
export RLINF_SPIDER_DATA_DIR="/mnt/public/yule/data/spider"
# spider -> agentlightning -> agent -> examples -> rlinf 根目录
REPO_PATH=$(dirname "$(dirname "$(dirname "$(dirname "$CONFIG_PATH")")")")
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${REPO_PATH}/examples:${PYTHONPATH}

if [[ "$1" == eval=* ]] || [[ "$1" == +eval=* ]]; then
    CONFIG_NAME="qwen2.5-1.5b-coder-enginehttp"
    ARGS=()
    for arg in "$@"; do
        if [[ "$arg" == eval=* ]] && [[ "$arg" != +eval=* ]]; then
            ARGS+=("+eval=true")
        elif [[ "$arg" == eval_checkpoint_dir=* ]] && [[ "$arg" != +eval_checkpoint_dir=* ]]; then
            ARGS+=("+${arg}")
        else
            ARGS+=("$arg")
        fi
    done
    ARGS+=("+runner.resume_dir=null")
    python "${CONFIG_PATH}/main.py" --config-path "${CONFIG_PATH}/config/" --config-name "${CONFIG_NAME}" "${ARGS[@]}"
else
    if [ -z "${1:-}" ]; then
        CONFIG_NAME="qwen2.5-1.5b-coder-enginehttp"
    else
        CONFIG_NAME="$1"
        shift
    fi
    python "${CONFIG_PATH}/main.py" --config-path "${CONFIG_PATH}/config/" --config-name "${CONFIG_NAME}" "$@"
fi
