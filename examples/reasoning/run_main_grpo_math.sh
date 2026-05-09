#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:$PYTHONPATH
export PYTHONPATH=/mnt/project_rlinf/yuanqwang/params_resharding_release-1022:$PYTHONPATH
# export PYTHONPATH=/mnt/public/yuanqwang/schedule-paper/ElasticMegatron:$PYTHONPATH
# export NCCL_DEBUG=INFO
# bash examples/reasoning/run_main_grpo_math.sh qwen2.5-1.5b-grpo-megatron-dynamic-1node

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-1.5b-grpo-megatron"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/reasoning/main_grpo.py --config-path ${CONFIG_PATH}/config/math/  --config-name $CONFIG_NAME