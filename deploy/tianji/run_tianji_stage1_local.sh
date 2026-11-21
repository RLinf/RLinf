#!/usr/bin/env bash
# Frozen Stage1 model + manual-start HIL on this machine.
# Usage: run_tianji_stage1_local.sh [--subtask-prompts subtasks.json] [Stage1 options]
set -euo pipefail
local_launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
stage1_args=()
while (( $# )); do
  case "$1" in
    --subtask-prompts) export TIANJI_SUBTASK_PROMPTS="$2"; shift 2 ;;
    --subtask-prompts=*) export TIANJI_SUBTASK_PROMPTS="${1#*=}"; shift ;;
    *) stage1_args+=("$1"); shift ;;
  esac
done
# shellcheck disable=SC1091
source "$local_launcher/tianji_local.env"
bash "$local_launcher/check_tianji_local.sh" --mode stage1 --live
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  inventory="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)"
  CUDA_VISIBLE_DEVICES="$(printf '%s\n' "$inventory" | "$TIANJI_CPU_PYTHON" "$local_launcher/select_stage1_gpu.py" "${TIANJI_STAGE1_GPU:-auto}")"
  export CUDA_VISIBLE_DEVICES
fi
export TIANJI_STAGE1_OUTPUT="${TIANJI_STAGE1_OUTPUT:-${TIANJI_RUNS_DIR}/stage1_$(date +%Y%m%d_%H%M%S)_$$}"
echo "STAGE1_LOCAL output=$TIANJI_STAGE1_OUTPUT gpu=$CUDA_VISIBLE_DEVICES"
if [[ "${TIANJI_ACTION_SPACE:-eef}" == joint ]]; then
  exec bash "$local_launcher/run_tianji_stage1_joint.sh" "${stage1_args[@]}"
fi
exec bash "$local_launcher/run_tianji_stage1.sh" --mode execute \
  --control-profile wuji_compat --hil-port "$TIANJI_HIL_PORT" \
  --execute-steps "$TIANJI_ACTION_HORIZON" --prefetch-step "$TIANJI_STAGE1_PREFETCH_STEP" \
  ${TIANJI_SUBTASK_PROMPTS:+--subtask-prompts "$TIANJI_SUBTASK_PROMPTS"} \
  "${stage1_args[@]}"
