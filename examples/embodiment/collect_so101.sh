#!/usr/bin/env bash

set -euo pipefail

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$(dirname "${EMBODIED_PATH}")")"
PYTHON_BIN="${PYTHON_BIN:-${REPO_PATH}/.venv/bin/python}"
EPISODES="${1:-1}"

export EMBODIED_PATH REPO_PATH
export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

exec "${PYTHON_BIN}" "${EMBODIED_PATH}/collect_real_data.py" \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name realworld_so101_collect_data \
  "runner.num_data_episodes=${EPISODES}"
