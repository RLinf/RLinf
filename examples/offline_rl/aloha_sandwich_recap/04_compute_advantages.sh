#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_python
require_value_checkpoint
log "Value checkpoint ready: ${VALUE_CKPT}"

run_logged "compute_advantages" "${RUN_ROOT}/compute_advantages.log" \
  "${PY}" "${REPO}/examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py" \
  --config-path "${CONFIG_DIR}" \
  --config-name aloha_sandwich_recap_compute_advantages \
  advantage.value_checkpoint="${VALUE_CKPT}" \
  advantage.batch_size="${ADV_BATCH_SIZE}" \
  advantage.flush_interval="${ADV_FLUSH_INTERVAL}" \
  advantage.num_dataloader_workers_per_gpu=0 \
  advantage.prefetch_factor=2

if [[ ! -f "${ADVANTAGES}" ]]; then
  log "ERROR advantages sidecar missing: ${ADVANTAGES}"
  exit 1
fi
log "Advantages ready: ${ADVANTAGES}"
