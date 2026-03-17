#!/bin/bash
#
# test_desktop_beaker_connection.sh — Diagnose desktop <-> Beaker connectivity.
#
# Run this on the desktop machine after a Beaker job is up. It checks:
#   1. Local Tailscale availability
#   2. Desktop -> Beaker TCP reachability (SSH and Ray head ports)
#   3. SSH login to the Beaker container
#   4. Optional reverse-tunnel loopback (simulates RemoteEnv -> RobotServer path)
#
# Usage:
#   bash scripts/test_desktop_beaker_connection.sh --head-ip <tailscale-ip> [OPTIONS]

set -euo pipefail

HEAD_IP=""
REMOTE_USER="shiruic"
RAY_PORT=6379
SSH_PORT=22
LOCAL_TEST_PORT=50051
REMOTE_TEST_PORT=50052
CONNECT_TIMEOUT=5
TEST_REVERSE_TUNNEL=true

usage() {
    cat <<'EOF'
Usage: bash scripts/test_desktop_beaker_connection.sh --head-ip <IP> [OPTIONS]

Diagnose connectivity between the desktop and a Beaker container.

Required:
  --head-ip IP           Beaker container Tailscale IP from the job logs

Options:
  --remote-user USER     SSH user on the Beaker container (default: shiruic)
  --ray-port PORT        Ray head port to probe (default: 6379)
  --ssh-port PORT        SSH port to probe (default: 22)
  --local-port PORT      Local desktop port used for reverse tunnel test (default: 50051)
  --remote-port PORT     Remote Beaker port used for reverse tunnel test (default: 50052)
  --timeout SEC          TCP/SSH connect timeout in seconds (default: 5)
  --skip-reverse-tunnel  Skip the reverse SSH tunnel smoke test
  --help                 Show this help

Examples:
  bash scripts/test_desktop_beaker_connection.sh --head-ip 100.64.1.2

  bash scripts/test_desktop_beaker_connection.sh \
      --head-ip 100.64.1.2 \
      --remote-user shiruic \
      --local-port 50051 \
      --remote-port 50052
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)                usage ;;
        --head-ip)             HEAD_IP="$2"; shift 2 ;;
        --remote-user)         REMOTE_USER="$2"; shift 2 ;;
        --ray-port)            RAY_PORT="$2"; shift 2 ;;
        --ssh-port)            SSH_PORT="$2"; shift 2 ;;
        --local-port)          LOCAL_TEST_PORT="$2"; shift 2 ;;
        --remote-port)         REMOTE_TEST_PORT="$2"; shift 2 ;;
        --timeout)             CONNECT_TIMEOUT="$2"; shift 2 ;;
        --skip-reverse-tunnel) TEST_REVERSE_TUNNEL=false; shift ;;
        *)                     echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$HEAD_IP" ]; then
    echo "Error: --head-ip is required"
    echo ""
    usage
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required"
    exit 1
fi

if ! command -v ssh >/dev/null 2>&1; then
    echo "Error: ssh is required"
    exit 1
fi

if ! command -v tailscale >/dev/null 2>&1; then
    echo "Error: tailscale CLI not found on this machine"
    exit 1
fi

SSH_OPTS=(
    -o BatchMode=yes
    -o StrictHostKeyChecking=no
    -o ConnectTimeout="${CONNECT_TIMEOUT}"
    -p "${SSH_PORT}"
)

LOCAL_SERVER_PID=""
cleanup() {
    if [ -n "${LOCAL_SERVER_PID}" ]; then
        kill "${LOCAL_SERVER_PID}" 2>/dev/null || true
        wait "${LOCAL_SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

run_tcp_probe() {
    local host="$1"
    local port="$2"
    python3 - "$host" "$port" "$CONNECT_TIMEOUT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
timeout = float(sys.argv[3])

sock = socket.create_connection((host, port), timeout=timeout)
sock.close()
print(f"TCP OK: {host}:{port}")
PY
}

start_local_test_server() {
    python3 - "$LOCAL_TEST_PORT" <<'PY' &
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import sys

port = int(sys.argv[1])

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"desktop-beaker-ok\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        return

server = HTTPServer(("127.0.0.1", port), Handler)
server.serve_forever()
PY
    LOCAL_SERVER_PID=$!
}

echo "=== Desktop <-> Beaker Connectivity Check ==="
echo "Head IP:          ${HEAD_IP}"
echo "Remote user:      ${REMOTE_USER}"
echo "SSH port:         ${SSH_PORT}"
echo "Ray port:         ${RAY_PORT}"
echo "Reverse test:     ${TEST_REVERSE_TUNNEL}"
echo ""

echo "[1/5] Checking local Tailscale status..."
DESKTOP_IP="$(tailscale ip -4 2>/dev/null || true)"
if [ -z "${DESKTOP_IP}" ]; then
    echo "FAIL: tailscale is installed but no IPv4 Tailscale address was found"
    echo "Run: tailscale status"
    exit 1
fi
echo "PASS: Desktop Tailscale IP = ${DESKTOP_IP}"

echo ""
echo "[2/5] Probing desktop -> Beaker SSH TCP reachability..."
if run_tcp_probe "${HEAD_IP}" "${SSH_PORT}"; then
    echo "PASS: SSH port reachable"
else
    echo "FAIL: Could not open TCP connection to ${HEAD_IP}:${SSH_PORT}"
    echo "Check Beaker logs for the printed Tailscale IP and confirm the node appears in tailscale status"
    exit 1
fi

echo ""
echo "[3/5] Probing desktop -> Beaker Ray TCP reachability..."
if run_tcp_probe "${HEAD_IP}" "${RAY_PORT}"; then
    echo "PASS: Ray head port reachable"
else
    echo "FAIL: Could not open TCP connection to ${HEAD_IP}:${RAY_PORT}"
    echo "Ray may not have started yet. Check the Beaker logs for the Ray startup message."
    exit 1
fi

echo ""
echo "[4/5] Verifying SSH login..."
if ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${HEAD_IP}" "echo ssh-ok"; then
    echo "PASS: SSH login succeeded"
else
    echo "FAIL: SSH login to ${REMOTE_USER}@${HEAD_IP} failed"
    echo "Check your SSH key setup and whether the Beaker container is accepting SSH"
    exit 1
fi

if [ "${TEST_REVERSE_TUNNEL}" = false ]; then
    echo ""
    echo "[5/5] Reverse tunnel test skipped"
    exit 0
fi

echo ""
echo "[5/5] Testing reverse SSH tunnel (Beaker localhost -> Desktop localhost)..."
start_local_test_server
sleep 1

if ! kill -0 "${LOCAL_SERVER_PID}" 2>/dev/null; then
    echo "FAIL: Could not start local test server on 127.0.0.1:${LOCAL_TEST_PORT}"
    exit 1
fi

if ssh \
    "${SSH_OPTS[@]}" \
    -R "${REMOTE_TEST_PORT}:127.0.0.1:${LOCAL_TEST_PORT}" \
    "${REMOTE_USER}@${HEAD_IP}" \
    "python3 - '${REMOTE_TEST_PORT}' <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.create_connection(('127.0.0.1', port), timeout=5)
sock.sendall(b'GET / HTTP/1.0\\r\\nHost: localhost\\r\\n\\r\\n')
data = sock.recv(4096)
sock.close()
text = data.decode('utf-8', errors='replace')
if 'desktop-beaker-ok' not in text:
    raise SystemExit(f'unexpected response: {text!r}')
print('reverse-tunnel-ok')
PY"; then
    echo "PASS: Reverse SSH tunnel works"
    echo "The Beaker container can reach your desktop service through localhost:${REMOTE_TEST_PORT}"
else
    echo "FAIL: Reverse SSH tunnel smoke test failed"
    echo "Desktop -> Beaker TCP is fine, but Beaker could not loop back through the reverse tunnel"
    exit 1
fi

echo ""
echo "All connectivity checks passed."
