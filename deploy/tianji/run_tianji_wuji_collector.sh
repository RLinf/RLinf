#!/usr/bin/env bash
# Workstation only: the LoopInfra launcher owns both SDK endpoints.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
rlinf_root="${RLINF_ROOT:-$(cd -- "${script_dir}/../.." && pwd)}"

if [[ -n "${TIANJI_SITE_ENV:-}" && -f "${TIANJI_SITE_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${TIANJI_SITE_ENV}"
elif [[ -f "${script_dir}/site.env" ]]; then
  # shellcheck disable=SC1091
  source "${script_dir}/site.env"
elif [[ -f "${rlinf_root}/site.env" ]]; then
  # shellcheck disable=SC1091
  source "${rlinf_root}/site.env"
elif [[ -f "${HOME}/.config/tianji/site.env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.config/tianji/site.env"
fi

loopinfra_root="${LOOPINFRA_ROOT:-${HOME}/LoopInfra}"
runtime_dir="${TIANJI_RUNTIME_DIR:-${rlinf_root}/.runtime/tianji}"
runs_dir="${TIANJI_RUNS_DIR:-${TIANJI_RUNS_ROOT:-${rlinf_root}/runs}}"
config_root="${TIANJI_COLLECTOR_CONFIG_ROOT:-${runtime_dir}/collector}"
config_python="${TIANJI_CPU_PYTHON:-${RLINF_PYTHON:-python3}}"
wuji_speed="${TIANJI_WUJI_SPEED:-trial45}"
case "$wuji_speed" in
  trial45) ;;
  *) echo "TIANJI_WUJI_SPEED must be trial45 for this validated controller profile." >&2; exit 2 ;;
esac

sync_hz="${TIANJI_CONTROL_HZ:-30}"
case "$sync_hz" in
  10|15|30) ;;
  *) echo "TIANJI_CONTROL_HZ must be 10, 15 or 30 (LoopInfra sync tiers)." >&2; exit 2 ;;
esac

for arg in "$@"; do
  case "$arg" in
    --pico-*|--no-pico-*|--sync-frequency-hz|--sync-frequency-hz=*|--client-config*|--collection-config*)
      echo "This entry fixes the validated PICO profile and RLT configuration files." >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$loopinfra_root/scripts/tianji-collector.sh" ]]; then
  echo "LoopInfra collector not found under $loopinfra_root; set LOOPINFRA_ROOT." >&2
  exit 2
fi

want="$(realpath -e -- "$loopinfra_root")"
for pid in $(pgrep -u "$(id -u)" -f -- '-m loopinfra\.(client run|server --network-config)' || true); do
  have="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
  if [[ -n "$have" && "$have" != "$want" ]]; then
    echo "Running collector belongs to $have, expected $want; stop it before starting." >&2
    exit 2
  fi
done

reset_args=()
xense_args=(--enable-xense)
for arg in "$@"; do
  case "$arg" in --enable-xense|--disable-xense) xense_args=() ;; esac
done

if [[ -n "${TIANJI_RESET_POSE:-}" ]]; then
  reset_pose="$(realpath -e -- "$TIANJI_RESET_POSE")" || {
    echo "TIANJI_RESET_POSE not found: $TIANJI_RESET_POSE" >&2; exit 2; }
  reset_args=(--reset-pose "$reset_pose")
  export LOOPINFRA_RESET_POSE="$reset_pose"
else
  unset LOOPINFRA_RESET_POSE
fi

"$config_python" "$script_dir/prepare_tianji_collector.py" \
  "$loopinfra_root" "$config_root" "$runs_dir/collector" --sync-hz "$sync_hz" \
  ${reset_args[@]+"${reset_args[@]}"}

# Keep RLinf PYTHONPATH out of the LoopInfra collector processes.
exec env -u PYTHONPATH bash "$loopinfra_root/scripts/tianji-collector.sh" "${xense_args[@]}" "$@" \
  --client-config "$config_root/client.yaml" \
  --collection-config "$config_root/collection.yaml" \
  --sync-frequency-hz "$sync_hz" \
  --pico-backend-profile wuji_compat \
  --pico-feedback-recovery-mode feedback_brake_v1 \
  --pico-contact-assist \
  --pico-speed "$wuji_speed"
