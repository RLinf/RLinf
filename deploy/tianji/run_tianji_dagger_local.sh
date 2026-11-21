#!/usr/bin/env bash
# Dedicated DAgger data collection launcher using examples/embodiment/config/tianji_dagger_collect.yaml.
# Keeps frozen VLA execution active (deploy_actor_updates=0, RL updates disabled)
# while archiving complete episodes (hil_mode=dagger, episode_archive=true).
# Usage: run_tianji_dagger_local.sh [--no-session] [--subtask-prompts FILE] [Hydra overrides]
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: run_tianji_dagger_local.sh [--no-session] [--subtask-prompts FILE] [Hydra overrides]"
  echo "Runs DAgger data collection (hil_mode=dagger, episode_archive=true, deploy_actor_updates=false)."
  exit 0
fi

use_session=1
overrides=()
while (( $# )); do
  case "$1" in
    --no-session) use_session=0; shift ;;
    --subtask-prompts)
      if (( $# < 2 )); then echo "Missing --subtask-prompts value" >&2; exit 2; fi
      export TIANJI_SUBTASK_PROMPTS="$2"; shift 2 ;;
    --subtask-prompts=*)
      export TIANJI_SUBTASK_PROMPTS="${1#*=}"; shift ;;
    *) overrides+=("$1"); shift ;;
  esac
done

export TIANJI_STAGE2_CONFIG="tianji_dagger_collect"
export TIANJI_DEPLOY_ACTOR_UPDATES=0

# shellcheck disable=SC1091
source "$launcher/tianji_local.env"
preflight="$("$PYTHON_BIN" -B "$launcher/check_stage2_resume.py" --config "${overrides[@]}")"
printf '%s\n' "$preflight"
if [[ "$preflight" == *" tactile=true"* && "${TIANJI_STAGE2_OPTIONAL_XENSE:-1}" == 0 ]]; then
  export TIANJI_TACTILE_REQUIRED=1
else
  export TIANJI_TACTILE_REQUIRED=0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TIANJI_RLT_OUTPUT="${TIANJI_RLT_OUTPUT:-${TIANJI_RUNS_DIR}/dagger_$(date +%Y%m%d_%H%M%S)_$$}"
mkdir -p "$TIANJI_RLT_OUTPUT"

if (( use_session == 1 )); then
  exec bash "$launcher/start_tianji_stage2_async.sh" "${overrides[@]}"
fi

bash "$launcher/check_tianji_local.sh" --mode stage2-dagger --live
exec > >(tee -i -a "$TIANJI_RLT_OUTPUT/console.log") 2>&1
echo "DAGGER_LOCAL config=$TIANJI_STAGE2_CONFIG output=$TIANJI_RLT_OUTPUT gpu=$CUDA_VISIBLE_DEVICES"
exec bash "$launcher/run_tianji_stage2_async.sh" "${overrides[@]}"
