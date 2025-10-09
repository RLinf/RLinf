#! /bin/bash
set -x

tabs 4
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-1.5b-grpo-pipeline"
else
    CONFIG_NAME=$1
fi

python ${REPO_PATH}/examples/math/main_math.py --config-path $REPO_PATH/tests/e2e_tests/math/sglang  --config-name $CONFIG_NAME