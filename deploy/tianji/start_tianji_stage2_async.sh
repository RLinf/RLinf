#!/usr/bin/env bash
# Start/reuse the collector, wait for policy readiness, then run async Stage2.
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rlinf_root="${RLINF_ROOT:-$(cd -- "${launcher}/../.." && pwd)}"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo 'Usage: start_tianji_stage2_async.sh [--subtask-prompts FILE] [Hydra overrides]'
  echo 'Starts/reuses the collector and runs Stage2 async. Collector services remain running after training exits.'
  exit 0
fi
cd -- "$rlinf_root"

preflight_overrides=()
pending=("$@")
while (( ${#pending[@]} )); do
  case "${pending[0]}" in
    --subtask-prompts)
      if (( ${#pending[@]} < 2 )); then echo 'Missing --subtask-prompts value' >&2; exit 2; fi
      export TIANJI_SUBTASK_PROMPTS="${pending[1]}"; pending=("${pending[@]:2}") ;;
    --subtask-prompts=*)
      export TIANJI_SUBTASK_PROMPTS="${pending[0]#*=}"; pending=("${pending[@]:1}") ;;
    *) preflight_overrides+=("${pending[0]}"); pending=("${pending[@]:1}") ;;
  esac
done

# shellcheck disable=SC1091
source "$launcher/tianji_local.env"
export TIANJI_STAGE2_CONFIG="${TIANJI_STAGE2_CONFIG:-tianji2_rlt_stage2_async}"

preflight="$("$PYTHON_BIN" -B "$launcher/check_stage2_resume.py" --config "${preflight_overrides[@]}")"
printf '%s\n' "$preflight"
if [[ "$preflight" == *" tactile=true"* && "${TIANJI_STAGE2_OPTIONAL_XENSE:-1}" == 0 ]]; then
  export TIANJI_TACTILE_REQUIRED=1
else
  export TIANJI_TACTILE_REQUIRED=0
fi
if [[ "$preflight" == *" tactile=true"* && "$TIANJI_TACTILE_REQUIRED" == 0 ]]; then
  echo 'STAGE2_XENSE_OPTIONAL: starting with any number of online sensors; unavailable channels are masked.'
fi
if [[ "${TIANJI_PI05_SERVICE:-0}" != 0 ]]; then
  bash "$launcher/tianji_pi05_service.sh" ensure
fi
echo 'STAGE2_START: starting/checking collector (existing healthy services are reused).'
collector_args=()
if [[ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
  collector_args=(--no-browser)
  echo "STAGE2_START: no graphical session; collector page: ${LOOPINFRA_URL:-http://127.0.0.1:8780}/collector"
fi
bash "$launcher/run_tianji_wuji_collector.sh" ${collector_args[@]+"${collector_args[@]}"}
"$TIANJI_CPU_PYTHON" "$launcher/wait_tianji_collector.py" \
  --timeout "${TIANJI_COLLECTOR_READY_TIMEOUT:-120}"
echo 'STAGE2_START: collector ready; loading Stage2. The HIL browser will open after model readiness.'
if [[ "${TIANJI_PI05_SERVICE:-0}" != 0 ]]; then
  bash "$launcher/tianji_pi05_service.sh" wait
  echo 'STAGE2_START: pi05 service ready; the rollout will not load the VLA itself.'
fi
exec "$TIANJI_CPU_PYTHON" "$launcher/launch_tianji_stage2_session.py" "$@"
