#! /bin/bash
set -x

tabs 4
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:$PYTHONPATH
export LLMASJUDGE_API_URL=${LLMASJUDGE_API_URL:-"https://cloud.infini-ai.com/maas/v1/chat/completions"}
export LLMASJUDGE_API_KEY=${LLMASJUDGE_API_KEY:-"[your api key]"}
export LLMASJUDGE_MODEL=${LLMASJUDGE_MODEL:-"deepseek-v3.1"}

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-1.5b-grpo-llm_judge"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/coding_online_rl/main_coding_rl_llm_judge.py --config-path ${CONFIG_PATH}/config/  --config-name $CONFIG_NAME