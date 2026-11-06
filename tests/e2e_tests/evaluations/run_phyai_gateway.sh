#!/usr/bin/env bash
# Usage: GPU_ID=1 SERVER_GPU_ID=0 PHYAI_PYTHON=/venv/bin/python bash run_phyai_gateway.sh [SERVER_YAML]
# Each run owns its services and private Ray instance; existing clusters are untouched.
set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SERVER_CONFIG="${1:-$REPO_PATH/tests/e2e_tests/evaluations/phyai_gateway_server.yaml}"
PHYAI_PYTHON="$(command -v "${PHYAI_PYTHON:-python}")"
RLINF_VENV="${RLINF_VENV:-$REPO_PATH/.venv}"
GPU_ID="${GPU_ID:?Set GPU_ID to an available rendering GPU index}"
SERVER_GPU_ID="${SERVER_GPU_ID:-$GPU_ID}"
NUM_ENVS="${NUM_ENVS:-8}"
MAX_STEPS="${MAX_STEPS:-480}"
GATEWAY_PORT="${GATEWAY_PORT:-50111}"
HTTP_PORT="${HTTP_PORT:-30000}"
READY_TIMEOUT="${READY_TIMEOUT:-900}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
MIN_SUCCESS_ONCE="${MIN_SUCCESS_ONCE:-0}"

for tool in setsid timeout curl nvidia-smi; do
    command -v "$tool" >/dev/null || { printf 'Missing command: %s\n' "$tool" >&2; exit 1; }
done
[[ "$PHYAI_PYTHON" = /* ]] || PHYAI_PYTHON="$PWD/$PHYAI_PYTHON"
PHYAI_BIN="$(dirname "$PHYAI_PYTHON")"
[[ "$SERVER_CONFIG" = /* ]] || SERVER_CONFIG="$PWD/$SERVER_CONFIG"
[[ "$RLINF_VENV" = /* ]] || RLINF_VENV="$PWD/$RLINF_VENV"
[[ $# -le 1 ]] || { printf 'Expected at most one Server YAML argument\n' >&2; exit 1; }
[[ -f "$SERVER_CONFIG" && -f "$RLINF_VENV/bin/activate" && -x "$RLINF_VENV/bin/python" && -x "$PHYAI_PYTHON" ]] || {
    printf 'Server YAML or an executable Python environment is missing\n' >&2; exit 1;
}
for value in "$GPU_ID" "$SERVER_GPU_ID" "$NUM_ENVS" "$MAX_STEPS" "$READY_TIMEOUT" "$EVAL_TIMEOUT"; do
    [[ "$value" =~ ^(0|[1-9][0-9]*)$ ]] || { printf 'Invalid integer: %s\n' "$value" >&2; exit 1; }
done
(( NUM_ENVS > 0 && MAX_STEPS > 0 && MAX_STEPS % 16 == 0 && READY_TIMEOUT > 0 && EVAL_TIMEOUT > 0 )) || {
    printf 'Counts/timeouts must be positive; MAX_STEPS must be divisible by 16\n' >&2; exit 1;
}

cd "$REPO_PATH"
mkdir -p logs/ci
LOG_DIR="$(mktemp -d "$REPO_PATH/logs/ci/phyai-gateway.XXXXXX")"
printf 'Logs: %s\n' "$LOG_DIR"
MODEL_PORT="$("$PHYAI_PYTHON" - "$SERVER_CONFIG" "$NUM_ENVS" "$GATEWAY_PORT" "$HTTP_PORT" "$MIN_SUCCESS_ONCE" "$LOG_DIR/server.yaml" <<'PYTHON'
from pathlib import Path
import math
import os
import socket
import sys
import yaml

source = Path(sys.argv[1])
config = yaml.safe_load(source.read_text())
server = config["server"]
if server["model_name"] != "pi05" or server["host"] not in ("127.0.0.1", "0.0.0.0"):
    raise SystemExit("Server YAML must serve pi05 on a local IPv4 address")
if int(config["plugin_args"]["max_batch_size"]) < int(sys.argv[2]):
    raise SystemExit("Server max_batch_size must be at least NUM_ENVS")
for mapping, key, variable in [
    (config["plugin_args"], "checkpoint_dir", "PHYAI_HTTP_CHECKPOINT"),
    (config["adapter"]["args"], "tokenizer_dir", "PHYAI_HTTP_TOKENIZER"),
]:
    path = Path(os.environ.get(variable) or mapping[key]).expanduser()
    if not path.is_absolute():
        path = source.parent / path
    if not path.is_dir():
        raise SystemExit(f"Set {variable} to an existing resource directory, got {path}")
    mapping[key] = str(path.resolve())
if config.get("kernel_policy"):
    path = Path(config["kernel_policy"]).expanduser()
    config["kernel_policy"] = str((source.parent / path).resolve())
threshold = float(sys.argv[5])
if not math.isfinite(threshold) or not 0 <= threshold <= 1:
    raise SystemExit("MIN_SUCCESS_ONCE must be finite and in [0, 1]")
if os.environ.get("MODEL_PORT"):
    server["port"] = int(os.environ["MODEL_PORT"])
ports = [int(server["port"]), int(sys.argv[3]), int(sys.argv[4])]
if len(set(ports)) != 3 or any(not 1 <= port <= 65535 for port in ports):
    raise SystemExit("Server/Gateway/HTTP ports must be distinct and in 1..65535")
for port in ports:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
Path(sys.argv[6]).write_text(yaml.safe_dump(config, sort_keys=False))
print(ports[0])
PYTHON
)"
MODEL_ENDPOINT="127.0.0.1:$MODEL_PORT"
GATEWAY_URL="http://127.0.0.1:$HTTP_PORT"
pids=()

cleanup() {
    local status=$? alive pid i
    trap - EXIT INT TERM
    for (( i=${#pids[@]}-1; i>=0; i-- )); do
        kill -TERM -- "-${pids[i]}" 2>/dev/null || true
    done
    for (( i=0; i<10; i++ )); do
        alive=0
        for pid in "${pids[@]}"; do
            if kill -0 -- "-$pid" 2>/dev/null; then alive=1; fi
        done
        (( alive == 0 )) && break
        sleep 1
    done
    for pid in "${pids[@]}"; do
        kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    if (( status != 0 )); then
        for log in "$LOG_DIR"/*.log; do
            [[ -f "$log" ]] || continue
            printf '\n--- %s ---\n' "$log" >&2
            tail -n 40 "$log" >&2 || true
        done
    fi
    printf 'Logs retained in %s\n' "$LOG_DIR"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_ready() {
    local label=$1 deadline=$((SECONDS + READY_TIMEOUT)) pid
    shift
    while true; do
        for pid in "${pids[@]}"; do
            kill -0 "$pid" 2>/dev/null || { printf '%s process exited\n' "$label" >&2; return 1; }
        done
        if "$@" >/dev/null 2>&1; then
            printf '[ready] %s\n' "$label"
            return 0
        fi
        (( SECONDS < deadline )) || { printf 'Timed out waiting for %s\n' "$label" >&2; return 1; }
        sleep 1
    done
}
server_ready() {
    timeout 2 bash -c 'exec 3<>/dev/tcp/127.0.0.1/"$1"' _ "$MODEL_PORT"
}
backend_ready() {
    curl --noproxy '*' -fsS --max-time 2 "$GATEWAY_URL/v1/backends" |
        "$PHYAI_PYTHON" -c '
import json, sys
backends = json.load(sys.stdin)["backends"]
sys.exit(0 if any(b["model_name"] == "pi05" and b["endpoint"] == sys.argv[1]
                 and b["healthy"] is True for b in backends) else 1)
' "$MODEL_ENDPOINT"
}

export ROBOT_PLATFORM=LIBERO
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
export LIBERO_TYPE=standard
export PYTHONUNBUFFERED=1
nvidia-smi -i "$SERVER_GPU_ID,$GPU_ID" --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
printf 'Logs: %s\n' "$LOG_DIR"

service_env=(env "PATH=$PHYAI_BIN:$PATH")
if [[ -n "${PHYAI_CUDART_LIB_NAME:-}" ]]; then
    service_env+=("CUDNN_FRONTEND_CUDART_LIB_NAME=$PHYAI_CUDART_LIB_NAME")
fi
"${service_env[@]}" "CUDA_VISIBLE_DEVICES=$SERVER_GPU_ID" setsid "$PHYAI_PYTHON" -m phyai server "$LOG_DIR/server.yaml" >"$LOG_DIR/server.log" 2>&1 &
pids+=("$!")
wait_ready 'PI0.5 Server' server_ready

"${service_env[@]}" setsid "$PHYAI_PYTHON" -m phyai_gateway.server \
    --gateway "127.0.0.1:$GATEWAY_PORT" \
    --http-host 127.0.0.1 --http-port "$HTTP_PORT" \
    --backend pi05 "$MODEL_ENDPOINT" >"$LOG_DIR/gateway.log" 2>&1 &
pids+=("$!")
wait_ready 'Gateway backend healthy' backend_ready

RAY_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/phyai-http.XXXXXX")"
PATH="$RLINF_VENV/bin:$PATH" CUDA_VISIBLE_DEVICES="$GPU_ID" env -u RAY_ADDRESS setsid "$RLINF_VENV/bin/python" - "$LOG_DIR/ray-address" "$RAY_TMP_DIR" <<'PYTHON' >"$LOG_DIR/ray.log" 2>&1 &
from pathlib import Path
import signal
import sys
import time

import ray

stopping = False


def stop(signum, frame):
    global stopping
    stopping = True


try:
    context = ray.init(address="local", _temp_dir=sys.argv[2], include_dashboard=False)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    Path(sys.argv[1]).write_text(context.address_info["gcs_address"])
    while not stopping:
        time.sleep(0.2)
finally:
    ray.shutdown()
PYTHON
pids+=("$!")
ray_ready() { [[ -s "$LOG_DIR/ray-address" ]]; }
wait_ready 'private Ray cluster' ray_ready
RAY_ADDRESS="$(<"$LOG_DIR/ray-address")"
export RAY_ADDRESS

(
    source "$RLINF_VENV/bin/activate"
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
    exec setsid timeout --kill-after=30s "${EVAL_TIMEOUT}s" \
        bash evaluations/run_eval.sh libero libero_spatial_pi05_eval_phyai \
        "cluster.component_placement.env=$GPU_ID" \
        "cluster.component_placement.rollout=$GPU_ID" \
        "env.eval.total_num_envs=$NUM_ENVS" \
        env.eval.rollout_epoch=1 \
        "env.eval.max_episode_steps=$MAX_STEPS" \
        "env.eval.max_steps_per_rollout_epoch=$MAX_STEPS" \
        env.eval.video_cfg.save_video=false \
        "runner.logger.log_path='$LOG_DIR/rlinf'" \
        runner.logger.logger_backends=[] \
        rollout.rollout_backend=phyai \
        rollout.phyai.serving_mode=worker_http \
        "rollout.phyai.gateway_url=$GATEWAY_URL" \
        rollout.phyai.http_max_retries=0 \
        rollout.model.action_horizon=16 rollout.model.num_action_chunks=16
) >"$LOG_DIR/eval.log" 2>&1 &
pids+=("$!")
eval_pid=${pids[3]}
while kill -0 "$eval_pid" 2>/dev/null; do
    for pid in "${pids[@]:0:3}"; do
        kill -0 "$pid" 2>/dev/null || { printf 'Server, Gateway, or private Ray exited during evaluation\n' >&2; exit 1; }
    done
    sleep 1
done
wait "$eval_pid"

"$PHYAI_PYTHON" - "$LOG_DIR/rlinf/metrics.log" "$MIN_SUCCESS_ONCE" <<'PY' >"$LOG_DIR/result.json"
from pathlib import Path
import json
import math
import re
import sys

text = Path(sys.argv[1]).read_text()
metrics = {}
for name in ("num_trajectories", "success_once", "success_at_end"):
    matches = re.findall(rf"\b{name}\s*=\s*([^\s│]+)", text)
    if not matches:
        raise SystemExit(f"Missing final metric: {name}")
    metrics[name] = float(matches[-1])
if not all(math.isfinite(v) for v in metrics.values()) or metrics["num_trajectories"] <= 0:
    raise SystemExit(f"No valid completed trajectories: {metrics}")
if not all(0 <= metrics[name] <= 1 for name in ("success_once", "success_at_end")):
    raise SystemExit(f"Invalid success metrics: {metrics}")
if metrics["success_once"] < float(sys.argv[2]):
    raise SystemExit(f"success_once below {sys.argv[2]}: {metrics}")
print(json.dumps({"passed": True, "metrics": metrics}, indent=2))
PY
printf '[PASS] PhyAI Gateway evaluation completed; result: %s/result.json\n' "$LOG_DIR"
