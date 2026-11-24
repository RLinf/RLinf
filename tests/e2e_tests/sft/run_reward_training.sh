#! /bin/bash
set -x

tabs 4

REPO_PATH="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
export REPO_PATH

CONFIG="${1%.yaml}"
shift 1 2>/dev/null || true

export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

rlinf train --family reward "${REPO_PATH}/tests/e2e_tests/sft/${CONFIG}.yaml" "$@"
