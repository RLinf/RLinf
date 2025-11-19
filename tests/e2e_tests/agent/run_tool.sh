#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

python ${REPO_PATH}/examples/multiturn_demo/main_tool.py --config-path ${REPO_PATH}/tests/e2e_tests/agent  --config-name tool-qwen2.5-1.5b

