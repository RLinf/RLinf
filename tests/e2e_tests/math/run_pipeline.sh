#! /bin/bash
set -x

tabs 4
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

python ${REPO_PATH}/examples/math/main_math_pipeline.py --config-path $REPO_PATH/tests/e2e_tests/math  --config-name qwen2.5-1.5b-grpo-pipeline