#!/usr/bin/env bash
set -euo pipefail

# End-to-end ALOHA sandwich RECAP-style off-policy RL finetuning.
#
# Run the full pipeline:
#   bash examples/offline_rl/run_aloha_sandwich_recap.sh
#
# Run one step:
#   RUN_ROOT=/path/to/run bash examples/offline_rl/aloha_sandwich_recap/03_train_value_sft.sh
#
# Run in tmux:
#   tmux new-session -d -s aloha_recap -c /inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/RLinf \
#     'bash examples/offline_rl/run_aloha_sandwich_recap.sh'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_DIR="${SCRIPT_DIR}/aloha_sandwich_recap"
source "${STEP_DIR}/common.sh"

log_pipeline_header
require_policy_checkpoint

bash "${STEP_DIR}/01_convert_dataset.sh"
bash "${STEP_DIR}/02_compute_returns.sh"
bash "${STEP_DIR}/03_train_value_sft.sh"
bash "${STEP_DIR}/04_compute_advantages.sh"
bash "${STEP_DIR}/05_train_cfg_rl.sh"

log "ALOHA sandwich RECAP pipeline complete"
