#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

require_python
require_policy_checkpoint

CFG_COMMON=(
  "${PY}" "${REPO}/examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py"
  --config-path "${CONFIG_DIR}"
  --config-name aloha_sandwich_cfg_rl_openpi
  runner.logger.log_path="${CFG_RUN_DIR}"
  runner.max_steps="${CFG_STEPS}"
  runner.save_interval="${CFG_SAVE_INTERVAL:-600}"
  runner.val_check_interval=-1
  actor.model.model_path="${POLICY_CKPT}"
  data.num_workers=0
  actor.micro_batch_size="${CFG_MICRO_BATCH_SIZE:-64}"
  actor.global_batch_size="${CFG_GLOBAL_BATCH_SIZE:-512}"
)

set +e
run_logged "cfg_rl_full" "${RUN_ROOT}/cfg_rl_full.log" "${CFG_COMMON[@]}"
CFG_STATUS=$?
set -e

if [[ ${CFG_STATUS} -ne 0 ]] && grep -Eiq 'OutOfMemoryError|CUDA out of memory' "${RUN_ROOT}/cfg_rl_full.log"; then
  log "CFG full finetune OOM detected; retrying with LoRA runtime overrides"
  run_logged "cfg_rl_lora" "${RUN_ROOT}/cfg_rl_lora.log" \
    "${CFG_COMMON[@]}" \
    runner.logger.log_path="${CFG_RUN_DIR}_lora" \
    actor.model.is_lora=true \
    actor.model.lora_rank=32 \
    actor.micro_batch_size=1 \
    actor.global_batch_size=8 \
    actor.fsdp_config.gradient_checkpointing=true \
    actor.fsdp_config.use_orig_params=true
elif [[ ${CFG_STATUS} -ne 0 ]]; then
  log "ERROR cfg_rl_full failed without an OOM signature; see ${RUN_ROOT}/cfg_rl_full.log"
  exit "${CFG_STATUS}"
fi
