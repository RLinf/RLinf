#!/usr/bin/env bash
# Called by run_tianji_stage1_local.sh after collector/GPU preflight.
set -euo pipefail
entry_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$entry_dir/tianji_local.env"
if [[ "${TIANJI_ACTION_SPACE:-}" != joint ]]; then
  echo "This entry requires TIANJI_ACTION_SPACE=joint" >&2; exit 2
fi
overrides=()
while (( $# )); do
  case "$1" in
    --no-rtc-guidance) overrides+=(++env.eval.override_cfg.rtc_guidance=false); shift ;;
    --max-chunks)
      if [[ ! "${2:-}" =~ ^[0-9]+$ ]]; then echo "--max-chunks needs a nonnegative integer" >&2; exit 2; fi
      if (( $2 > 0 )); then overrides+=("env.eval.max_steps_per_rollout_epoch=$(( $2 * TIANJI_STAGE2_CHUNK ))"); fi
      shift 2 ;;
    --*) echo "Unsupported joint Stage1 option: $1; use Hydra overrides." >&2; exit 2 ;;
    *) overrides+=("$1"); shift ;;
  esac
done
export TIANJI_RLT_OUTPUT="${TIANJI_STAGE1_OUTPUT:-${TIANJI_RUNS_DIR}/stage1_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$TIANJI_RLT_OUTPUT"
# shellcheck disable=SC1091
source "$entry_dir/tianji_cuda_env.sh"
exec "$PYTHON_BIN" -u "$RLINF_ROOT/evaluations/eval_embodied_agent.py" \
  --config-path "$EMBODIED_PATH/config" --config-name tianji2_joint_stage1 "${overrides[@]}"
