#!/bin/bash
#
# start_robot_server.sh — Launch RobotServer + optional cloudflared tunnel.
#
# Usage:
#   bash scripts/start_robot_server.sh --config /path/to/yam_env.yaml [OPTIONS]
#
# Options:
#   --config PATH     Path to YAM env YAML config (required)
#   --port PORT       gRPC server port (default: 50051)
#   --tunnel          Start cloudflared tunnel for NAT traversal
#   --help            Show this help

set -euo pipefail

CONFIG=""
PORT=50051
TUNNEL=false
DUMMY=false

usage() {
    cat <<'EOF'
Usage: bash scripts/start_robot_server.sh --config PATH [OPTIONS]

Launch the gRPC RobotServer wrapping YAMEnv, with optional cloudflared tunnel.

Options:
  --config PATH     Path to YAM env YAML config (required)
  --port PORT       gRPC server port (default: 50051)
  --tunnel          Start cloudflared tunnel for NAT traversal
  --dummy           Run without real hardware (zero observations)
  --help            Show this help
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)    usage ;;
        --config)  CONFIG="$2"; shift 2 ;;
        --port)    PORT="$2"; shift 2 ;;
        --tunnel)  TUNNEL=true; shift ;;
        --dummy)   DUMMY=true; shift ;;
        *)         echo "Unknown option: $1"; usage ;;
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

if [ "$TUNNEL" = true ]; then
    echo "=== Starting cloudflared tunnel ==="
    if ! command -v cloudflared &>/dev/null; then
        echo "Error: cloudflared not found. Install from https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"
        exit 1
    fi
    cloudflared tunnel --url "localhost:${PORT}" &
    TUNNEL_PID=$!
    echo "Tunnel PID: ${TUNNEL_PID}"
    echo ""
    echo ">>> Set ROBOT_SERVER_URL to the tunnel URL printed above <<<"
    echo ""
fi

wait
