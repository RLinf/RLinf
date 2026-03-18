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
# Because Beaker jobs are ephemeral, the SSH host key behind "beaker-0" also
# changes across preemptions / restarts.  The tunnel therefore disables
# known_hosts persistence for this connection so stale host keys do not block
# reverse port forwarding on the desktop.
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
TUNNEL_PID=""
TUNNEL_LOG=""
TUNNEL_LAUNCHER=""

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

start_tunnel() {
    local ssh_args=(
        -N
        -R "${PORT}:localhost:${PORT}"
        -o ServerAliveInterval=10
        -o ServerAliveCountMax=3
        -o ExitOnForwardFailure=yes
        -o StrictHostKeyChecking=no
        -o UserKnownHostsFile=/dev/null
        -o GlobalKnownHostsFile=/dev/null
        -o ConnectTimeout=10
        -o BatchMode=yes
    )

    if command -v autossh &>/dev/null; then
        TUNNEL_LAUNCHER="autossh"
        export AUTOSSH_GATETIME=0
        autossh -M 0 "${ssh_args[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
            >"${TUNNEL_LOG}" 2>&1 &
    else
        TUNNEL_LAUNCHER="ssh"
        echo "WARNING: autossh not found; falling back to plain ssh." | tee -a "${TUNNEL_LOG}"
        echo "         The tunnel will work, but it will not auto-reconnect if it drops." | tee -a "${TUNNEL_LOG}"
        ssh "${ssh_args[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
            >>"${TUNNEL_LOG}" 2>&1 &
    fi

    TUNNEL_PID=$!
}

print_tunnel_failure_hint() {
    echo ""
    echo "ERROR: ${TUNNEL_LAUNCHER} (PID ${TUNNEL_PID}) died within 3 seconds."
    echo "  Try manually:  ssh -v -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o GlobalKnownHostsFile=/dev/null ${REMOTE_USER}@${REMOTE_HOST} echo ok"
    echo "  Common causes: host unreachable, SSH key rejected, autossh not installed"
    echo "  Tunnel log: ${TUNNEL_LOG}"
    if [ -s "${TUNNEL_LOG}" ]; then
        echo "  Last tunnel log lines:"
        tail -n 20 "${TUNNEL_LOG}" | sed 's/^/    /'
    fi
    echo ""
    echo "RobotServer is still running (PID ${SERVER_PID}). Tunnel is NOT active."
}

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
    TUNNEL_LOG=$(mktemp "/tmp/rlinf_robot_tunnel.${PORT}.XXXX.log")
    echo "=== Starting persistent reverse SSH tunnel ==="
    echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}"
    echo "Tunnel: ${REMOTE_HOST}:localhost:${PORT} -> this machine:${PORT}"
    echo "Log:    ${TUNNEL_LOG}"
    echo ""
    echo "autossh will reconnect automatically when a new Beaker job starts."
    echo "If autossh is unavailable, the script falls back to plain ssh."
    echo "ROBOT_SERVER_URL=localhost:${PORT} is set in submit_yam_training.sh."
    echo ""

    start_tunnel
    echo "${TUNNEL_LAUNCHER} PID: ${TUNNEL_PID}"

    # Give the tunnel a moment to start, then verify it's still alive.
    sleep 3
    if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
        print_tunnel_failure_hint
    else
        echo "${TUNNEL_LAUNCHER} is running."
        echo ""
    fi
fi

# Monitor both processes: warn if the tunnel dies while robot server is still up.
while true; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "RobotServer (PID ${SERVER_PID}) exited."
        break
    fi
    if [ "$NO_TUNNEL" = false ] && [ -n "${TUNNEL_PID:-}" ]; then
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            echo ""
            echo "WARNING: ${TUNNEL_LAUNCHER} (PID ${TUNNEL_PID}) died. Restarting tunnel..."
            start_tunnel
            echo "Restarted ${TUNNEL_LAUNCHER} with PID ${TUNNEL_PID}"
            sleep 1
            if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
                print_tunnel_failure_hint
            fi
        fi
    fi
    sleep 10
done

wait
