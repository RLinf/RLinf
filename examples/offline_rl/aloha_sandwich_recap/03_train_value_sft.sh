#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_python

run_logged "value_sft_${VALUE_STEPS}" "${RUN_ROOT}/value_sft.log" \
  "${PY}" "${REPO}/examples/offline_rl/advantage_labeling/recap/train_value.py" \
  --config-path "${CONFIG_DIR}" \
  --config-name aloha_sandwich_recap_value_model_sft \
  runner.logger.log_path="${VALUE_RUN_DIR}" \
  runner.max_steps="${VALUE_STEPS}" \
  runner.save_interval="${VALUE_STEPS}" \
  runner.val_check_interval=-1 \
  actor.micro_batch_size="${VALUE_MICRO_BATCH_SIZE}" \
  actor.global_batch_size="${VALUE_GLOBAL_BATCH_SIZE}" \
  data.train_num_workers=0 \
  data.eval_num_workers=0 \
  data.prefetch_factor=null \
  data.persistent_workers=false

require_value_checkpoint
log "Value checkpoint ready: ${VALUE_CKPT}"
