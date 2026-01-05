#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export RAY_DEBUG=1

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${REPO_PATH}/examples:$PYTHONPATH


python ${REPO_PATH}/examples/mas/main_eval.py --config-path ${CONFIG_PATH}/config/  --config-name "mas_eval_qwen2.5_7b_format_yushi"

python ${REPO_PATH}/examples/mas/main_eval.py --config-path ${CONFIG_PATH}/config/  --config-name "mas_eval_qwen2.5_7b_format_yushi_widesearch"

