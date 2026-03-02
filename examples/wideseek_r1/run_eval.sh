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

if [ -z "$1" ]; then
    CONFIG_NAME="eval_qwen3_qa"
    # qwen2.5-3b-searchr1_eval
else
    CONFIG_NAME=$1
fi


python ${REPO_PATH}/examples/wideseek_r1/eval.py \
    --config-path ${CONFIG_PATH}/config/ \
    --config-name $CONFIG_NAME \
    "runner.experiment_name=train_mas_model_final" \
    "agentloop.workflow=mas" \
    "rollout.model.model_path=/mnt/project_rlinf/xzxuan/wideseek_model/mas_hybrid_final" \



python ${REPO_PATH}/examples/wideseek_r1/eval.py \
    --config-path ${CONFIG_PATH}/config/ \
    --config-name $CONFIG_NAME \
    "runner.experiment_name=train_sa_model_final" \
    "agentloop.workflow=sa" \
    "rollout.model.model_path=/mnt/project_rlinf/xzxuan/wideseek_model/sa_hybrid_final" \

