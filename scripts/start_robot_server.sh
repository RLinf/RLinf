#!/bin/bash
#
# start_robot_server.sh — Launch RobotServer + persistent reverse SSH tunnel to Beaker.
#
# The robot desktop can reach the Beaker container via Tailscale, but not
# the other way around.  A reverse SSH tunnel exposes the local gRPC port
# on the container so that RemoteEnv can connect to localhost:<port>.
#
# Uses autossh to keep the tunnel alive across Beaker job restarts.
# Every Beaker job registers the Tailscale hostname "beaker-0", so the
# tunnel automatically reconnects when a new job starts — no IP needed.
#
# RemoteEnv retries the gRPC connection for grpc_connect_timeout seconds
# (default 300s), giving the tunnel time to establish after job submission.
#
# Usage:
#   bash scripts/start_robot_server.sh --config /path/to/yam_env.yaml [OPTIONS]
#
# Options:
#   --config PATH         Path to YAM env YAML config (required)
#   --port PORT           gRPC server port (default: 50051)
#   --remote-host HOST    Beaker Tailscale hostname or IP (default: beaker-0)
#   --remote-user USER    SSH user on the Beaker container (default: shiruic)
#   --no-tunnel           Start RobotServer only, no SSH tunnel
#   --dummy               Run without real hardware (zero observations)
#   --help                Show this help

set -euo pipefail

CONFIG=""
PORT=50051
REMOTE_HOST="beaker-0"
REMOTE_USER="shiruic"
NO_TUNNEL=false
DUMMY=false

usage() {
    cat <<'EOF'
Usage: bash scripts/start_robot_server.sh --config PATH [OPTIONS]

Launch the gRPC RobotServer wrapping YAMEnv, with a persistent autossh reverse
tunnel to Beaker. The tunnel reconnects automatically when a new Beaker job
starts (all jobs register the Tailscale hostname "beaker-0").

Options:
  --config PATH         Path to YAM env YAML config (required)
  --port PORT           gRPC server port (default: 50051)
  --remote-host HOST    Beaker Tailscale hostname or IP (default: beaker-0)
  --remote-user USER    SSH user on the Beaker container (default: shiruic)
  --no-tunnel           Start RobotServer only, without SSH tunnel
  --dummy               Run without real hardware (zero observations)
  --help                Show this help

Examples:
  # Persistent server + auto-reconnecting tunnel (default beaker-0 hostname):
  bash scripts/start_robot_server.sh --config examples/embodiment/config/env/yam.yaml

  # Local only (no tunnel, for testing):
  bash scripts/start_robot_server.sh --config examples/embodiment/config/env/yam.yaml \
      --no-tunnel --dummy

  # Explicit IP instead of hostname (e.g. for one-off debugging):
  bash scripts/start_robot_server.sh --config examples/embodiment/config/env/yam.yaml \
      --remote-host 100.87.5.72
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)         usage ;;
        --config)       CONFIG="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --remote-host)  REMOTE_HOST="$2"; shift 2 ;;
        --remote-user)  REMOTE_USER="$2"; shift 2 ;;
        --no-tunnel)    NO_TUNNEL=true; shift ;;
        --dummy)        DUMMY=true; shift ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$CONFIG" ]; then
    echo "Error: --config is required"
    exit 1
fi

cleanup() {
    echo "Shutting down..."
    kill 0 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "=== Resetting CAN interfaces ==="
bash YAM/yam_realtime/yam_realtime/scripts/reset_all_can.sh
echo ""

echo "=== Starting RobotServer ==="
echo "Config: ${CONFIG}"
echo "Port:   ${PORT}"
echo ""

SERVER_ARGS=(--config-path "${CONFIG}" --port "${PORT}")
[ "$DUMMY" = true ] && SERVER_ARGS+=(--dummy)

python -m rlinf.envs.remote.robot_server "${SERVER_ARGS[@]}" &
SERVER_PID=$!

if [ "$NO_TUNNEL" = false ]; then
    if ! command -v autossh &>/dev/null; then
        echo "Error: autossh not found. Install with:"
        echo "  sudo apt-get install autossh   # Ubuntu/Debian"
        echo "  brew install autossh           # macOS"
        exit 1
    fi

    echo "=== Starting persistent reverse SSH tunnel ==="
    echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}"
    echo "Tunnel: ${REMOTE_HOST}:localhost:${PORT} -> this machine:${PORT}"
    echo ""
    echo "autossh will reconnect automatically when a new Beaker job starts."
    echo "ROBOT_SERVER_URL=localhost:${PORT} is set in submit_yam_training.sh."
    echo ""

    # -M 0:  disable autossh's own monitoring port; rely on SSH keepalives instead
    # -N:    no remote command, tunnel only
    # -R:    reverse tunnel — Beaker localhost:PORT -> this machine localhost:PORT
    # ServerAliveInterval/CountMax: SSH detects dead connection within 30s
    # ExitOnForwardFailure: SSH exits if the tunnel can't be bound (triggers autossh retry)
    # StrictHostKeyChecking=no: Beaker containers have a fresh host key each run
    autossh -M 0 -N \
        -R "${PORT}:localhost:${PORT}" \
        -o ServerAliveInterval=10 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        "${REMOTE_USER}@${REMOTE_HOST}" &
    AUTOSSH_PID=$!
    echo "autossh PID: ${AUTOSSH_PID}"
    echo ""
fi

wait
