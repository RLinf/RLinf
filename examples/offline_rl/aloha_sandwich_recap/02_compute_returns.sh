#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_python

if [[ ! -f "${RETURNS}" ]]; then
  run_logged "compute_returns" "${RUN_ROOT}/compute_returns.log" \
    "${PY}" "${REPO}/examples/offline_rl/advantage_labeling/recap/process/compute_returns.py" \
    --config-path "${CONFIG_DIR}" \
    --config-name aloha_sandwich_recap_compute_returns
else
  log "SKIP compute_returns: ${RETURNS} already exists"
fi
