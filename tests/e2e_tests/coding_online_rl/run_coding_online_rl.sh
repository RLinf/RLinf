#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

python ${REPO_PATH}/examples/coding_online_rl/main_coding_online_rl.py --config-path ${REPO_PATH}/tests/e2e_tests/coding_online_rl  --config-name qwen2.5-1.5b-ppo

