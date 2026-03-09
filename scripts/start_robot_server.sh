#!/bin/bash
#
# start_robot_server.sh — Launch RobotServer + reverse SSH tunnel to Beaker.
#
# The robot desktop can reach the Beaker container via Tailscale, but not
# the other way around.  A reverse SSH tunnel exposes the local gRPC port
# on the container so that RemoteEnv can connect to localhost:<port>.
#
# Usage:
#   bash scripts/start_robot_server.sh --config /path/to/yam_env.yaml [OPTIONS]
#
# Options:
#   --config PATH         Path to YAM env YAML config (required)
#   --port PORT           gRPC server port (default: 50051)
#   --remote-host HOST    Beaker container Tailscale IP (for reverse SSH tunnel)
#   --remote-user USER    SSH user on the Beaker container (default: current user)
#   --dummy               Run without real hardware (zero observations)
#   --help                Show this help

set -euo pipefail

CONFIG=""
PORT=50051
REMOTE_HOST=""
REMOTE_USER="shiruic"
DUMMY=false

usage() {
    cat <<'EOF'
Usage: bash scripts/start_robot_server.sh --config PATH [OPTIONS]

Launch the gRPC RobotServer wrapping YAMEnv, with optional reverse SSH tunnel
to a Beaker container over Tailscale.

Options:
  --config PATH         Path to YAM env YAML config (required)
  --port PORT           gRPC server port (default: 50051)
  --remote-host HOST    Beaker container Tailscale IP (for reverse SSH tunnel)
  --remote-user USER    SSH user on the Beaker container (default: current user)
  --dummy               Run without real hardware (zero observations)
  --help                Show this help

Examples:
  # Local only (no tunnel):
  bash scripts/start_robot_server.sh --config examples/embodiment/config/env/yam.yaml --dummy

  # With reverse SSH tunnel to Beaker container:
  bash scripts/start_robot_server.sh --config examples/embodiment/config/env/yam.yaml \
      --remote-host 100.87.5.72 --remote-user shiruic --dummy
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

echo "=== Starting RobotServer ==="
echo "Config: ${CONFIG}"
echo "Port:   ${PORT}"
echo ""

SERVER_ARGS=(--config-path "${CONFIG}" --port "${PORT}")
[ "$DUMMY" = true ] && SERVER_ARGS+=(--dummy)

python -m rlinf.envs.remote.robot_server "${SERVER_ARGS[@]}" &
SERVER_PID=$!

if [ -n "$REMOTE_HOST" ]; then
    echo "=== Setting up reverse SSH tunnel ==="
    echo "Forwarding ${REMOTE_HOST}:localhost:${PORT} -> localhost:${PORT}"
    echo ""
    # -N: no remote command, -R: reverse tunnel, -o: keep alive
    ssh -N \
        -R "${PORT}:localhost:${PORT}" \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        "${REMOTE_USER}@${REMOTE_HOST}" &
    SSH_PID=$!
    echo "SSH tunnel PID: ${SSH_PID}"
    echo ""
    echo ">>> On the Beaker container, set: export ROBOT_SERVER_URL=localhost:${PORT} <<<"
    echo ""
fi

wait
