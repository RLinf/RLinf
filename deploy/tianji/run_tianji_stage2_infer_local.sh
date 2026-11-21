#!/usr/bin/env bash
# Inference only: frozen Stage1 + trained Stage2 actor, no optimizer worker.
# Usage: run_tianji_stage2_infer_local.sh CHECKPOINT [--subtask-prompts subtasks.json] [Hydra overrides]
set -euo pipefail
entry_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (( $# < 1 )); then
  echo "Usage: run_tianji_stage2_infer_local.sh /absolute/path/to/actor/model_state_dict/full_weights.pt [--subtask-prompts subtasks.json] [Hydra overrides]" >&2
  exit 2
fi
checkpoint="$(realpath -- "$1")"
shift
overrides=()
while (( $# )); do
  case "$1" in
    --subtask-prompts) export TIANJI_SUBTASK_PROMPTS="$2"; shift 2 ;;
    --subtask-prompts=*) export TIANJI_SUBTASK_PROMPTS="${1#*=}"; shift ;;
    *) overrides+=("$1"); shift ;;
  esac
done
# shellcheck disable=SC1091
source "$entry_dir/tianji_local.env"
if [[ ! -f "$checkpoint" ]]; then
  echo "Stage2 actor checkpoint does not exist: $checkpoint" >&2
  exit 2
fi
preflight="$("$PYTHON_BIN" -B "$entry_dir/check_stage2_infer.py" "$checkpoint" "${overrides[@]}")"
printf '%s\n' "$preflight"
case "$preflight" in
  *" tactile=true"*) export TIANJI_TACTILE_REQUIRED=1 ;;
  *) export TIANJI_TACTILE_REQUIRED=0 ;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$entry_dir/check_tianji_local.sh" --mode stage2-infer --live
export TIANJI_RLT_OUTPUT="${TIANJI_RLT_OUTPUT:-${TIANJI_RUNS_DIR}/stage2_infer_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$TIANJI_RLT_OUTPUT"
# shellcheck disable=SC1091
source "$entry_dir/tianji_cuda_env.sh"
exec "$PYTHON_BIN" -u "$RLINF_ROOT/evaluations/eval_embodied_agent.py" \
  --config-path "$EMBODIED_PATH/config" --config-name tianji2_rlt_stage2_infer \
  runner.ckpt_path="$checkpoint" "${overrides[@]}"
