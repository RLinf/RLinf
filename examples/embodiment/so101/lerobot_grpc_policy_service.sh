#!/usr/bin/env bash
set -euo pipefail

service_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd "${service_dir}/../../.." && pwd)"
config_file="${SO101_GRPC_CONFIG:-${service_dir}/lerobot_grpc_policy.env}"
[[ -f "${config_file}" ]] || {
  echo "Copy ${service_dir}/lerobot_grpc_policy.env.example to ${config_file} and edit it." >&2
  exit 2
}
# shellcheck disable=SC1090
source "${config_file}"
: "${SO101_GRPC_CHECKPOINT:?Missing SO101_GRPC_CHECKPOINT}"
: "${SO101_GRPC_NORM_STATS:?Missing SO101_GRPC_NORM_STATS}"
: "${SO101_GRPC_HOST:=0.0.0.0}"
: "${SO101_GRPC_PORT:=50051}"
: "${SO101_GRPC_CUDA_VISIBLE_DEVICES:=0}"
: "${SO101_GRPC_DEVICE:=cuda:0}"
: "${SO101_GRPC_FPS:=15}"
: "${SO101_GRPC_TMUX_SESSION:=so101_lerobot_grpc}"
: "${SO101_GRPC_LOG:=${root_dir}/examples/embodiment/so101/lerobot_grpc_policy.log}"
: "${RLINF_PYTHON:=python}"

is_listening() {
  ss -ltn 2>/dev/null | awk -v port=":${SO101_GRPC_PORT}" '$4 ~ (port "$" ) {found=1} END {exit !found}'
}

check_assets() {
  [[ -s "${SO101_GRPC_CHECKPOINT}/model_state_dict/full_weights.pt" ]] || {
    echo "Checkpoint weights are missing: ${SO101_GRPC_CHECKPOINT}" >&2; exit 3;
  }
  [[ -s "${SO101_GRPC_NORM_STATS}" ]] || {
    echo "Norm stats are missing: ${SO101_GRPC_NORM_STATS}" >&2; exit 3;
  }
}

serve_foreground() {
  check_assets
  mkdir -p "$(dirname "${SO101_GRPC_LOG}")"
  cd "${root_dir}"
  exec env PYTHONPATH="${root_dir}${PYTHONPATH:+:${PYTHONPATH}}" \
    CUDA_VISIBLE_DEVICES="${SO101_GRPC_CUDA_VISIBLE_DEVICES}" "${RLINF_PYTHON}" \
    "${service_dir}/lerobot_grpc_policy_server.py" \
    --checkpoint "${SO101_GRPC_CHECKPOINT}" --norm-stats "${SO101_GRPC_NORM_STATS}" \
    --host "${SO101_GRPC_HOST}" --port "${SO101_GRPC_PORT}" --fps "${SO101_GRPC_FPS}" \
    --device "${SO101_GRPC_DEVICE}" >>"${SO101_GRPC_LOG}" 2>&1
}

start() {
  check_assets
  if tmux has-session -t "${SO101_GRPC_TMUX_SESSION}" 2>/dev/null; then return 0; fi
  tmux new-session -d -s "${SO101_GRPC_TMUX_SESSION}" "${service_dir}/lerobot_grpc_policy_service.sh _serve"
}
stop() { tmux kill-session -t "${SO101_GRPC_TMUX_SESSION}" 2>/dev/null || true; }
status() {
  tmux has-session -t "${SO101_GRPC_TMUX_SESSION}" 2>/dev/null && echo "tmux=running" || echo "tmux=stopped"
  is_listening && echo "socket=listening endpoint=${SO101_GRPC_HOST}:${SO101_GRPC_PORT}" || echo "socket=not-listening endpoint=${SO101_GRPC_HOST}:${SO101_GRPC_PORT}"
}
case "${1:-}" in
  start) start ;; stop) stop ;; restart) stop; start ;; status) status ;; logs) tail -n "${SO101_GRPC_LOG_LINES:-100}" "${SO101_GRPC_LOG}" ;; _serve) serve_foreground ;;
  *) echo "Usage: $0 {start|stop|restart|status|logs}" >&2; exit 2 ;;
esac
