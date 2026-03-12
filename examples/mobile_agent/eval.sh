#!/bin/bash

set -e

set -x
export CUDA_VISIBLE_DEVICES=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

export TOKENIZERS_PARALLELISM=false

export RAY_DEDUP_LOGS=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"

RLINF_PATH="${PROJECT_ROOT}/RLinf"

export PYTHONPATH="${PROJECT_ROOT}:${RLINF_PATH}:${PYTHONPATH}"

if [ -z "$1" ]; then
    CONFIG_NAME="qwen3vl-4b-eval"
else
    CONFIG_NAME="$1"
fi

python3 "${SCRIPT_DIR}/main.py" --config-name "${CONFIG_NAME}"