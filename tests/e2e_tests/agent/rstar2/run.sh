#! /bin/bash
set -x

tabs 4

REPO_PATH="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
export REPO_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

CONFIG_NAME="${1:-qwen2.5-1.5b-megatron}"
CONFIG_NAME="${CONFIG_NAME%.yaml}"
if [ "$#" -gt 0 ]; then
    shift
fi

rlinf train --family agent --subgroup rstar2 "${REPO_PATH}/tests/e2e_tests/agent/rstar2/${CONFIG_NAME}.yaml" "$@"
