#! /bin/bash
set -x

tabs 4

REPO_PATH="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export REPO_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "$1" ]; then
    echo "Please provide a config name as the first argument."
    exit 1
else
    CONFIG_NAME="${1%.yaml}"
    shift
fi

rlinf train --family reasoning "${REPO_PATH}/tests/e2e_tests/dynamic_scheduler/${CONFIG_NAME}.yaml" "$@"
