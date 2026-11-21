#!/usr/bin/env bash
# Resident pi05 inference service for Stage2 (systemd user unit tianji-pi05).
#
#   tianji_pi05_service.sh ensure    start if absent; restart if identity differs
#   tianji_pi05_service.sh wait      block until ready and matching
#   tianji_pi05_service.sh start|stop|restart|status|info|check|identity|logs
set -euo pipefail
launcher="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cmd="${1:-status}"
unit=tianji-pi05
module=rlinf.models.embodiment.openpi_rlinf.pi05_service

# shellcheck disable=SC1091
source "$launcher/tianji_local.env"
export TIANJI_STAGE2_CONFIG="${TIANJI_STAGE2_CONFIG:-tianji2_rlt_stage2_async}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# shellcheck disable=SC1091
source "$launcher/tianji_cuda_env.sh"
runtime="${TIANJI_RUNTIME_DIR:-$RLINF_ROOT/.runtime/tianji}"
mkdir -p "$runtime"
frames="${TIANJI_PI05_FRAMES:-$launcher/pi05_selfcheck_frames.npz}"
logfile="$runtime/tianji-pi05.log"
sock="${TIANJI_PI05_SOCKET:-${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/tianji-pi05.sock}"

say() { printf '%s PI05_SERVICE: %s\n' "$(date +%T)" "$*"; }

service_env_args() {
  local name
  while IFS= read -r name; do
    printf -- '--setenv=%s\n' "$name"
  done < <(env | grep -E '^(TIANJI_|RLT_|RLINF_|LOOPINFRA_|PYTHONPATH=|EMBODIED_PATH=|OPENPI_DATA_HOME=|HF_HOME=|JAX_PLATFORMS=|XLA_PYTHON_CLIENT_PREALLOCATE=|USE_TF=|TOKENIZERS_PARALLELISM=|OMP_NUM_THREADS=|TMPDIR=|LD_LIBRARY_PATH=|PYTHONDONTWRITEBYTECODE=|PATH=|HOME=|XDG_RUNTIME_DIR=)' | grep -v '^TIANJI_FAST_INFER=' | cut -d= -f1)
}

do_start() {
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  if systemctl --user is-active --quiet "$unit"; then
    say "unit $unit is already active"
    return 0
  fi
  local args=()
  [[ -f "$frames" ]] && args+=(--frames "$frames")
  mapfile -t envs < <(service_env_args)
  say "starting $unit on GPU0 (log: $logfile)"
  systemd-run --user --unit="$unit" --description="Tianji Stage2 resident pi05 inference (GPU0)" \
    --working-directory="$RLINF_ROOT" --collect --quiet \
    -p MemoryMax="${TIANJI_PI05_MEMORY_MAX:-12G}" -p MemorySwapMax=0 \
    -p StandardOutput=append:"$logfile" -p StandardError=append:"$logfile" \
    --setenv=CUDA_VISIBLE_DEVICES=0 "${envs[@]}" \
    "$PYTHON_BIN" -u -m "$module" serve --device cuda:0 --fast "${TIANJI_PI05_FAST:-1}" \
    ${TIANJI_PI05_VERIFY_HASH:+--verify-hash} "${args[@]}"
}

do_stop() {
  if systemctl --user is-active --quiet "$unit"; then
    say "stopping $unit"
    systemctl --user stop "$unit"
  else
    say "unit $unit is not active"
  fi
  systemctl --user reset-failed "$unit" 2>/dev/null || true
  for _ in $(seq 20); do [[ -S "$sock" ]] || break; sleep 0.5; done
  [[ -S "$sock" ]] && rm -f "$sock"
  if command -v nvidia-smi >/dev/null 2>&1; then
    say "GPU0 now: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader -i 0 2>/dev/null || echo '?')"
  fi
}

case "$cmd" in
  start) do_start ;;
  stop) do_stop ;;
  restart) do_stop; do_start ;;
  ensure)
    set +e
    out="$("$PYTHON_BIN" -m "$module" check 2>&1)"; rc=$?
    set -e
    case "$rc" in
      0) say "reusing the running service ($(printf '%s' "$out" | tail -1))" ;;
      3) say "no service running; starting it"; do_start ;;
      4) say "NOTICE: the running service differs from this tree/config -> restarting it"
         printf '%s\n' "$out" | sed 's/^/    /'
         do_stop; do_start ;;
      5) say "service still loading: $(printf '%s' "$out" | tail -1)" ;;
      6) say "the service failed earlier ($(printf '%s' "$out" | tail -1)); starting it again"; do_stop; do_start ;;
      *) say "unexpected check result $rc:"; printf '%s\n' "$out"; exit "$rc" ;;
    esac ;;
  wait)
    set +e
    "$PYTHON_BIN" -m "$module" wait --timeout "${TIANJI_PI05_READY_TIMEOUT:-600}"; rc=$?
    set -e
    if (( rc != 0 )); then
      say "service not usable (exit $rc); last log lines:"
      tail -n 15 "$logfile" 2>/dev/null | sed 's/^/    /' || true
      exit "$rc"
    fi ;;
  status)
    systemctl --user status --no-pager "$unit" 2>&1 | head -12 || true
    set +e; "$PYTHON_BIN" -m "$module" check; echo "check exit $?"; set -e ;;
  info|check|identity) exec "$PYTHON_BIN" -m "$module" "$cmd" ;;
  logs) exec tail -n "${2:-50}" "$logfile" ;;
  *) echo "Usage: tianji_pi05_service.sh ensure|wait|start|stop|restart|status|info|check|identity|logs" >&2; exit 2 ;;
esac
