#!/bin/bash
# setup_franka_collect.sh - Startup script for data collection (single node).
# Usage:
#   source setup_franka_collect.sh           # Activate the control env (ROS + catkin + venv)
#   source setup_franka_collect.sh start     # Activate env and start a single-node Ray head (rank 0)
#   source setup_franka_collect.sh stop      # Stop Ray on this node
#
# Data collection runs only on the control node (rank 0) without the compute
# node, so this script starts an independent single-node Ray head. It does not
# depend on the compute environment.
#
# Note: no `set -u` here - ROS catkin profile scripts may exit on unbound
# variables (e.g. ROS_DISTRO).
#
# This script is independent of setup_franka_node.sh (which is used to join the
# compute head for two-node training).
# Cleanup of stale processes is handled by cleanup.sh.
#
# Required environment variables:
#   FRANKA_ROBOT_IP  Robot IP (required with `start`).
#   RLINF_HEAD_IP    Reachable IP for this Ray head (required with `start`).
# Optional environment variables:
#   RLINF_VENV, RLINF_COMM_NET_DEVICES, RAY_TEMP_DIR

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "$_SCRIPT_DIR")")"

# ---- Environment paths and role ----
export RLINF_VENV="${RLINF_VENV:-${REPO_PATH}/.venv}"   # control env (contains franka_catkin_ws / ROS / libfranka)
export RLINF_NODE_RANK=0                                 # the control node is the only rank 0 during collection
# Set RLINF_COMM_NET_DEVICES to the network interface that other nodes can reach.
export RLINF_COMM_NET_DEVICES="${RLINF_COMM_NET_DEVICES:-eth0}"

# ---- Ray configuration ----
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rlinf_collect}"   # independent temp-dir (isolated from two-node mode)

if [ "${1:-}" = "start" ] && [ -z "${FRANKA_ROBOT_IP:-}" ]; then
  echo "[setup_franka_collect.sh] ERROR: FRANKA_ROBOT_IP is required with 'start'." >&2
  echo "  export FRANKA_ROBOT_IP=<robot_ip_address>" >&2
  unset _SCRIPT_DIR
  return 1 2>/dev/null || exit 1
fi
if [ "${1:-}" = "start" ] && [ -z "${RLINF_HEAD_IP:-}" ]; then
  echo "[setup_franka_collect.sh] ERROR: RLINF_HEAD_IP is required with 'start'." >&2
  echo "  export RLINF_HEAD_IP=<this_node_reachable_ip>" >&2
  unset _SCRIPT_DIR
  return 1 2>/dev/null || exit 1
fi
if [ -n "${FRANKA_ROBOT_IP:-}" ]; then
  export FRANKA_ROBOT_IP
fi
if [ -n "${RLINF_HEAD_IP:-}" ]; then
  export RLINF_HEAD_IP
  export RAY_ADDRESS="${RLINF_HEAD_IP}:6379"
fi

# Clean up environment variables that may interfere with Python.
unset PYTHONHOME

if [ ! -f "${RLINF_VENV}/bin/activate" ]; then
  echo "[setup_franka_collect.sh] ERROR: venv not found: ${RLINF_VENV}/bin/activate" >&2
  echo "  cd ${REPO_PATH} && bash requirements/install.sh embodied --env franka --venv ${RLINF_VENV}" >&2
  return 1 2>/dev/null || exit 1
fi

# Activate the control env (source ROS + catkin first, then the venv so the
# venv's Python overrides ROS's Python).
if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
else
  echo "[setup_franka_collect.sh] WARNING: ROS setup.bash not found: /opt/ros/noetic/setup.bash" >&2
fi
CATKIN_SETUP="${RLINF_VENV}/franka_catkin_ws/devel/setup.bash"
if [ -f "${CATKIN_SETUP}" ]; then
  source "${CATKIN_SETUP}"
else
  echo "[setup_franka_collect.sh] WARNING: catkin workspace not found: ${CATKIN_SETUP}" >&2
fi

source "${RLINF_VENV}/bin/activate"
hash -r

# Put this repo first on PYTHONPATH.
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

# Default single-machine ROS configuration.
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
if [ -z "${ROS_IP:-}" ]; then
  export ROS_IP="$(python -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0]); s.close()" 2>/dev/null || echo 127.0.0.1)"
fi

echo "[setup_franka_collect.sh] role: collect (single-node control rank 0)"
echo "[setup_franka_collect.sh] repo: ${REPO_PATH}"
echo "[setup_franka_collect.sh] venv: ${RLINF_VENV}"
echo "[setup_franka_collect.sh] robot_ip: ${FRANKA_ROBOT_IP}"
echo "[setup_franka_collect.sh] python: $(which python)"

# Verify that the key control library can be imported.
python - <<'PY' || echo "[setup_franka_collect.sh] WARNING: franka_gripper import failed (fix before starting the controller)"
try:
    import franka_gripper.msg
    print("[setup_franka_collect.sh] franka_gripper import ok")
except Exception as e:
    print(f"[setup_franka_collect.sh] franka_gripper import failed: {e}")
    raise
PY

if [ "${1:-}" = "start" ]; then
  # Start a single-node Ray head.
  mkdir -p "${RAY_TEMP_DIR}"
  ray start --head --port=6379 --node-ip-address="${RLINF_HEAD_IP}" \
    --temp-dir="${RAY_TEMP_DIR}"
  sleep 2

  echo "[setup_franka_collect.sh] Single-node Ray head started @ ${RAY_ADDRESS}"
  echo "[setup_franka_collect.sh] Now run: bash examples/embodiment/collect_data.sh"
elif [ "${1:-}" = "stop" ]; then
  ray stop
fi

unset _SCRIPT_DIR
