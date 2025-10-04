#! /bin/bash
set -x

tabs 4
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-vl-3b-grpo-collocated"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/reasoning/main_grpo.py --config-path $REPO_PATH/tests/e2e_tests/vqa/sglang  --config-name $CONFIG_NAME