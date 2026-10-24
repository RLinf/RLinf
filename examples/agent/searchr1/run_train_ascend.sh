#!/bin/bash
set -x

export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export HCCL_INTRA_ROCE_ENABLE=1

CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "$(dirname "$CONFIG_PATH")")")"
MEGATRON_PATH="${MEGATRON_PATH:-/opt/Megatron-LM}"
export PYTHONPATH="${REPO_PATH}:${MEGATRON_PATH}:${REPO_PATH}/examples:${PYTHONPATH:-}"

# Enable multi-process execution on a single NPU.
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
# Let Ray preserve the Ascend visible-device environment.
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

CONFIG_NAME="${1:-train_deepseek_r1_ascend}"

python "${REPO_PATH}/examples/agent/searchr1/train.py" \
    --config-path "${CONFIG_PATH}/config/" \
    --config-name "${CONFIG_NAME}"
