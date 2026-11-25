#! /bin/bash
set -euo pipefail
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$(cd "$CONFIG_PATH/../../.." && pwd)"
export PYTHONPATH=${REPO_PATH}:${REPO_PATH}/examples${PYTHONPATH:+:$PYTHONPATH}
if [ -n "${MEGATRON_PATH:-}" ]; then
    export PYTHONPATH=${MEGATRON_PATH}:${PYTHONPATH}
fi

if [ -z "${1:-}" ]; then
    CONFIG_NAME="train_qwen3.5_4b_episode_sync_grpo"
else
    CONFIG_NAME=$1
    shift
fi

exec python "${REPO_PATH}/examples/agent/rpent/train.py" --config-path "${CONFIG_PATH}/config/"  --config-name "$CONFIG_NAME" "$@"
