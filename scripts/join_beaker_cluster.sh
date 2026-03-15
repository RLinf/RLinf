#!/bin/bash
#
# join_beaker_cluster.sh — Join a Beaker Ray cluster from the local desktop and run training.
#
# This is the second half of a desktop-driven training workflow:
#   1. submit_yam_beaker_cluster.sh starts a Beaker job with Ray head + GPUs (idle).
#   2. This script joins the cluster from the desktop and runs training locally.
#
# NOTE: The canonical YAM configs (yam_ppo_openpi, yam_ppo_openpi_topreward) use
# env/remote_yam (RemoteEnv via gRPC) and cluster.num_nodes: 1.  For those configs,
# use submit_yam_training.sh instead — it runs everything on Beaker.
#
# This script is useful when you have a custom config with env/yam (direct YAMEnv)
# and a multi-node cluster layout that puts the env worker on the desktop.  In that
# case the desktop joins as a Ray worker node and the training script drives the env
# directly without gRPC.
#
# Usage:
#   bash scripts/join_beaker_cluster.sh --head-ip <tailscale-ip> [OPTIONS] [-- HYDRA_OVERRIDES...]

set -euo pipefail

# --- Defaults ---
HEAD_IP=""
CONFIG_NAME="yam_ppo_openpi"
MODEL_PATH=""
TASK_DESC="pick and place"
NODE_RANK=1
RAY_PORT=6379
EXTRA_OVERRIDES=()

usage() {
    cat <<'EOF'
Usage: bash scripts/join_beaker_cluster.sh --head-ip <IP> [OPTIONS] [-- HYDRA_OVERRIDES...]

Join a Beaker Ray cluster from the local desktop and run training.

The training script runs locally. The config's native placement puts actor/rollout
on the Beaker GPUs and the env worker on this desktop (with direct YAMEnv access).

Required:
  --head-ip IP          Beaker container Tailscale IP (from Beaker logs)

Options:
  --config NAME         Hydra config name (default: yam_ppo_openpi)
  --model-path PATH     Model checkpoint path
  --task DESC           Task description (default: "pick and place")
  --node-rank N         Desktop node rank (default: 1)
  --ray-port PORT       Ray port (default: 6379)
  --help                Show this help

Extra Hydra overrides can be passed after '--':
  bash scripts/join_beaker_cluster.sh --head-ip 100.x.y.z -- algorithm.update_epoch=2

Examples:
  # Custom 2-node config with env/yam (desktop at rank 1, Beaker GPUs at rank 0)
  bash scripts/join_beaker_cluster.sh \
      --head-ip 100.64.1.2 \
      --config my_custom_yam_config \
      --node-rank 1 \
      --model-path thomas0829/folding_towel_pi05 \
      --task "pick and place"
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help)         usage ;;
        --head-ip)      HEAD_IP="$2"; shift 2 ;;
        --config)       CONFIG_NAME="$2"; shift 2 ;;
        --model-path)   MODEL_PATH="$2"; shift 2 ;;
        --task)         TASK_DESC="$2"; shift 2 ;;
        --node-rank)    NODE_RANK="$2"; shift 2 ;;
        --ray-port)     RAY_PORT="$2"; shift 2 ;;
        --)             shift; EXTRA_OVERRIDES=("$@"); break ;;
        *)              echo "Unknown option: $1"; usage ;;
    esac
done

if [ -z "$HEAD_IP" ]; then
    echo "Error: --head-ip is required"
    echo ""
    usage
fi

# --- Detect entry script ---
ENTRY_SCRIPT="train_embodied_agent.py"
case "$CONFIG_NAME" in
    *topreward*|*staged*|yam_ppo_openpi)
        ENTRY_SCRIPT="train_embodied_agent_staged.py"
        ;;
esac

# --- Cleanup trap: stop local Ray worker on exit ---
cleanup() {
    echo ""
    echo "Stopping local Ray worker..."
    ray stop --force 2>/dev/null || true
}
trap cleanup EXIT

# --- Set node rank and join Ray cluster ---
export RLINF_NODE_RANK="$NODE_RANK"

# Ray requires this env var to allow non-Linux nodes to join a cluster.
export RAY_ENABLE_WINDOWS_OR_OSX_CLUSTER=1

echo "=== Joining Beaker Ray Cluster ==="
echo "Head IP:      ${HEAD_IP}"
echo "Ray port:     ${RAY_PORT}"
echo "Node rank:    ${NODE_RANK}"
echo "Config:       ${CONFIG_NAME}"
echo "Entry script: ${ENTRY_SCRIPT}"
echo "Model:        ${MODEL_PATH:-<not set>}"
echo "Task:         ${TASK_DESC}"
echo ""

# Use the Tailscale IP as the node IP so the GCS health checks reach the
# correct interface. Auto-detection may pick a local IP Beaker cannot reach.
DESKTOP_IP=$(tailscale ip -4 2>/dev/null || true)
if [ -n "$DESKTOP_IP" ]; then
    echo "Using Tailscale IP for Ray node: ${DESKTOP_IP}"
    RAY_JOIN_ARGS=(--address="${HEAD_IP}:${RAY_PORT}" --node-ip-address="${DESKTOP_IP}")
else
    echo "WARNING: Could not detect Tailscale IP — Ray will auto-detect node IP"
    RAY_JOIN_ARGS=(--address="${HEAD_IP}:${RAY_PORT}")
fi

# --- TCP pre-check ---
echo "Checking TCP connectivity to ${HEAD_IP}:${RAY_PORT}..."
if python3 -c "
import socket, sys
try:
    s = socket.create_connection(('${HEAD_IP}', ${RAY_PORT}), timeout=5)
    s.close()
    print('TCP connection OK')
except Exception as e:
    print(f'TCP connection FAILED: {e}')
    sys.exit(1)
" 2>&1; then
    echo "Port ${RAY_PORT} reachable — proceeding."
else
    echo ""
    echo "Error: Cannot reach ${HEAD_IP}:${RAY_PORT} via TCP."
    echo "  - Verify the Beaker container Tailscale IP and that it's in your Tailscale network"
    echo "  - Check Beaker logs confirm Ray head started (look for 'Ray runtime started')"
    echo "  - Run: tailscale status  (to verify the container appears as a peer)"
    exit 1
fi

echo "Connecting to Ray head at ${HEAD_IP}:${RAY_PORT}..."
for i in $(seq 1 30); do
    echo "  ray start attempt ${i}/30..."
    if ray start "${RAY_JOIN_ARGS[@]}"; then
        echo "Successfully joined Ray cluster"
        break
    fi
    if [ "$i" = "30" ]; then
        echo "Error: Could not join Ray cluster after 30 attempts"
        exit 1
    fi
    echo "  Retrying in 10s..."
    sleep 10
done

# --- Verify raylet stays alive after joining ---
# Wait 15 seconds and confirm the local raylet socket still exists.
# Active health checks from Beaker → desktop don't work with userspace
# Tailscale (no kernel TUN interface). Node death detection relies on
# heartbeats (desktop → Beaker), which do work.
echo "Waiting 15s to verify local raylet stays alive..."
sleep 15
SESSION_LATEST=$(readlink -f /tmp/ray/session_latest 2>/dev/null || echo "")
if [ -n "$SESSION_LATEST" ] && [ -S "${SESSION_LATEST}/sockets/raylet" ]; then
    echo "Raylet is alive."
else
    echo "ERROR: Local raylet socket is gone — raylet died within 15s of joining."
    echo "Check: cat /tmp/ray/session_latest/logs/raylet.out | tail -20"
    exit 1
fi

# --- Activate .venv if present ---
if [ -f ".venv/bin/activate" ]; then
    echo "Activating .venv in $(pwd)"
    source .venv/bin/activate
fi

echo "Python: $(which python) ($(python --version 2>&1))"

# Install deps if not already present.
if ! python -c "import hydra" 2>/dev/null; then
    echo "Dependencies not found — running: uv sync --extra embodied"
    uv sync --extra embodied
fi

# --- Build and run the training command ---
TRAIN_CMD="python examples/embodiment/${ENTRY_SCRIPT}"
TRAIN_CMD+=" --config-name ${CONFIG_NAME}"

if [ -n "$MODEL_PATH" ]; then
    TRAIN_CMD+=" actor.model.model_path=${MODEL_PATH}"
    TRAIN_CMD+=" rollout.model.model_path=${MODEL_PATH}"
fi

TRAIN_CMD+=" 'env.train.task_description=${TASK_DESC}'"
TRAIN_CMD+=" 'env.eval.task_description=${TASK_DESC}'"

for override in "${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"}"; do
    TRAIN_CMD+=" ${override}"
done

export EMBODIED_PATH="examples/embodiment"

echo ""
echo "Running training command:"
echo "  ${TRAIN_CMD}"
echo ""

eval "$TRAIN_CMD"
