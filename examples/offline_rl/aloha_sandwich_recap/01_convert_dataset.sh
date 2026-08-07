#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_python

if [[ ! -f "${DATASET}/meta/info.json" ]]; then
  run_logged "convert_aloha_hdf5_to_lerobot_v21" "${RUN_ROOT}/convert_aloha_hdf5_to_lerobot_v21.log" \
    "${PY}" "${REPO}/examples/offline_rl/data/convert_aloha_hdf5_to_lerobot_v21.py" \
    --raw-dir "${RAW_DATA}" \
    --output-dir "${DATASET}" \
    --repo-id "aloha/sandwich_rl" \
    --task "sandwich" \
    --fps 25
else
  log "SKIP convert_aloha_hdf5_to_lerobot_v21: ${DATASET}/meta/info.json already exists"
fi
