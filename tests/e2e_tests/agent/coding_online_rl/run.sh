#! /bin/bash
set -x

tabs 4

REPO_PATH="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
export REPO_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

rlinf train --family agent --subgroup coding_online_rl "${REPO_PATH}/tests/e2e_tests/agent/coding_online_rl/qwen2.5-1.5b-ppo.yaml" "$@"
