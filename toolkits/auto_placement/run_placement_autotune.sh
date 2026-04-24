#! /bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export REPO_PATH
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

CONFIG_NAME="${1:-maniskill_ppo_openvla_quickstart}"

python "${REPO_PATH}/toolkits/auto_placement/auto_placement_worker.py" \
  --config-path "${EMBODIED_PATH}/config" \
  --config-name "${CONFIG_NAME}"
