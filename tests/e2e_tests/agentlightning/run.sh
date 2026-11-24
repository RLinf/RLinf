#! /bin/bash
set -x

unset https_proxy HTTPS_PROXY http_proxy HTTP_PROXY

tabs 4

REPO_PATH="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export REPO_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

rlinf train --family agent --subgroup calc_x "${REPO_PATH}/tests/e2e_tests/agentlightning/qwen2.5-1.5b-multiturn.yaml" "$@"
