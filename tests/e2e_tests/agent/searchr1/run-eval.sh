#! /bin/bash
set -x

tabs 4

REPO_PATH="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
export REPO_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "$1" ] || [[ "$1" == *=* ]] || [[ "$1" == +* ]] || [[ "$1" == ~* ]] || [[ "$1" == -* ]]; then
    CONFIG_NAME=qwen2.5-1.5b-eval
else
    CONFIG_NAME="${1%.yaml}"
    shift
fi

rlinf eval --family agent --subgroup searchr1 "${REPO_PATH}/tests/e2e_tests/agent/searchr1/${CONFIG_NAME}.yaml" "$@"
