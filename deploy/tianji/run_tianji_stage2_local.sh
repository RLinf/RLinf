#!/usr/bin/env bash
# Stage2 policy inference + HIL learning on this machine (async: GPU0 rollout, GPU1 training).
# Usage: run_tianji_stage2_local.sh async [--subtask-prompts subtasks.json] [Hydra overrides]
set -euo pipefail
local_launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-async}"
if (( $# )); then shift; fi
case "$mode" in
  async) ;;
  dagger)
    exec bash "$local_launcher/run_tianji_dagger_local.sh" "$@"
    ;;
  sync)
    echo "Stage2 sync training was removed; use: run_tianji_stage2_local.sh async [options]" >&2
    exit 2
    ;;
  *)
    echo "Usage: run_tianji_stage2_local.sh async|dagger [--subtask-prompts subtasks.json] [Hydra overrides]" >&2
    exit 2
    ;;
esac

overrides=()
while (( $# )); do
  case "$1" in
    --subtask-prompts) export TIANJI_SUBTASK_PROMPTS="$2"; shift 2 ;;
    --subtask-prompts=*) export TIANJI_SUBTASK_PROMPTS="${1#*=}"; shift ;;
    *) overrides+=("$1"); shift ;;
  esac
done

# shellcheck disable=SC1091
source "$local_launcher/tianji_local.env"
export TIANJI_STAGE2_CONFIG="${TIANJI_STAGE2_CONFIG:-tianji2_rlt_stage2_async}"

preflight="$("$PYTHON_BIN" -B "$local_launcher/check_stage2_resume.py" --config "${overrides[@]}")"
printf '%s\n' "$preflight"
if [[ "$preflight" == *" tactile=true"* && "${TIANJI_STAGE2_OPTIONAL_XENSE:-1}" == 0 ]]; then
  export TIANJI_TACTILE_REQUIRED=1
else
  export TIANJI_TACTILE_REQUIRED=0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
bash "$local_launcher/check_tianji_local.sh" --mode stage2-async --live
export TIANJI_RLT_OUTPUT="${TIANJI_RLT_OUTPUT:-${TIANJI_RUNS_DIR}/stage2_async_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$TIANJI_RLT_OUTPUT"
exec > >(tee -i -a "$TIANJI_RLT_OUTPUT/console.log") 2>&1
echo "STAGE2_LOCAL mode=async config=$TIANJI_STAGE2_CONFIG output=$TIANJI_RLT_OUTPUT gpu=$CUDA_VISIBLE_DEVICES console=$TIANJI_RLT_OUTPUT/console.log"
if [[ -n "${TIANJI_STAGE2_SHADOW_VARIANT:-}" ]]; then
  exec "$PYTHON_BIN" -B -m rlinf.tianji_shadow supervise "$TIANJI_STAGE2_SHADOW_VARIANT" \
    bash "$local_launcher/run_tianji_stage2_async.sh" "${overrides[@]}"
fi
exec bash "$local_launcher/run_tianji_stage2_async.sh" "${overrides[@]}"
