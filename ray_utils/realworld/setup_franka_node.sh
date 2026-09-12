#!/bin/bash
# setup_franka_node.sh - Startup script for the Franka control node (Ray worker).
# Usage:
#   source setup_franka_node.sh          # Activate the control env (ROS + catkin + venv)
#   source setup_franka_node.sh start    # Activate env and join the compute head as rank 1
#   source setup_franka_node.sh stop     # Stop Ray on this node
#
# This node only serves the Franka control role (env / controller worker) and
# joins the Ray head as rank 1. Start the compute node (rank 0) via
# setup_compute_node.sh before joining it.
#
# Required environment variables:
#   FRANKA_ROBOT_IP  Robot IP (required with `start`).
#   RLINF_HEAD_IP    Compute Ray head IP (required with `start`).
# Optional environment variables:
#   RLINF_VENV, RLINF_COMM_NET_DEVICES, RAY_TEMP_DIR, RAY_ADDRESS

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "$_SCRIPT_DIR")")"

# Control environment. Defaults to the repository's .venv
# (contains franka_catkin_ws / ROS / libfranka).
export RLINF_VENV="${RLINF_VENV:-${REPO_PATH}/.venv}"
# The control node is fixed as rank 1 (joins the compute head).
export RLINF_NODE_RANK=1
# Set RLINF_COMM_NET_DEVICES to the network interface that other nodes can reach.
export RLINF_COMM_NET_DEVICES="${RLINF_COMM_NET_DEVICES:-eth0}"

# Ray uses an independent temp dir - it must differ from the compute head's so
# that this node is recognized as a separate node. Use a short path under /tmp
# to stay within the 107-byte AF_UNIX socket length limit.
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rlinf_control}"

if [ "${1:-}" = "start" ] && [ -z "${FRANKA_ROBOT_IP:-}" ]; then
  echo "[setup_franka_node.sh] ERROR: FRANKA_ROBOT_IP is required with 'start'." >&2
  echo "  export FRANKA_ROBOT_IP=<robot_ip_address>" >&2
  unset _SCRIPT_DIR
  return 1 2>/dev/null || exit 1
fi
if [ "${1:-}" = "start" ] && [ -z "${RLINF_HEAD_IP:-}" ]; then
  echo "[setup_franka_node.sh] ERROR: RLINF_HEAD_IP is required with 'start'." >&2
  echo "  export RLINF_HEAD_IP=<compute_head_ip_address>" >&2
  unset _SCRIPT_DIR
  return 1 2>/dev/null || exit 1
fi
if [ -n "${FRANKA_ROBOT_IP:-}" ]; then
  export FRANKA_ROBOT_IP
fi
if [ -n "${RLINF_HEAD_IP:-}" ]; then
  export RLINF_HEAD_IP
  export RAY_ADDRESS="${RAY_ADDRESS:-${RLINF_HEAD_IP}:6379}"
fi

# Clean up environment variables that may interfere with Python.
unset PYTHONHOME

if [ ! -f "${RLINF_VENV}/bin/activate" ]; then
  echo "[setup_franka_node.sh] ERROR: venv not found: ${RLINF_VENV}/bin/activate" >&2
  echo "  cd ${REPO_PATH} && bash requirements/install.sh embodied --env franka --venv ${RLINF_VENV}" >&2
  unset _SCRIPT_DIR
  return 1 2>/dev/null || exit 1
fi

# Source ROS and catkin first, then activate the venv so that the venv's Python
# overrides ROS's Python.
if [ -f /opt/ros/noetic/setup.bash ]; then
  source /opt/ros/noetic/setup.bash
else
  echo "[setup_franka_node.sh] WARNING: ROS setup.bash not found: /opt/ros/noetic/setup.bash" >&2
fi
CATKIN_SETUP="${RLINF_VENV}/franka_catkin_ws/devel/setup.bash"
if [ -f "${CATKIN_SETUP}" ]; then
  source "${CATKIN_SETUP}"
else
  echo "[setup_franka_node.sh] WARNING: catkin workspace not found: ${CATKIN_SETUP}" >&2
fi

source "${RLINF_VENV}/bin/activate"
hash -r

# Put this repo first on PYTHONPATH; keep the ROS package paths added by catkin.
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

# Default single-machine ROS configuration.
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://127.0.0.1:11311}"
if [ -z "${ROS_IP:-}" ]; then
  export ROS_IP="$(python -c "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.connect(('8.8.8.8',80)); print(s.getsockname()[0]); s.close()" 2>/dev/null || echo 127.0.0.1)"
fi

echo "[setup_franka_node.sh] role: control/franka"
echo "[setup_franka_node.sh] repo: ${REPO_PATH}"
echo "[setup_franka_node.sh] venv: ${RLINF_VENV}"
echo "[setup_franka_node.sh] node_rank: ${RLINF_NODE_RANK}"
echo "[setup_franka_node.sh] robot_ip: ${FRANKA_ROBOT_IP}"
echo "[setup_franka_node.sh] python: $(which python)"

# Verify that the key control library can be imported.
python - <<'PY' || echo "[setup_franka_node.sh] WARNING: franka_gripper import failed (fix before starting the controller)"
try:
    import franka_gripper.msg
    print("[setup_franka_node.sh] franka_gripper import ok")
except Exception as e:
    print(f"[setup_franka_node.sh] franka_gripper import failed: {e}")
    raise
PY

# Join the compute head as rank 1. Use an independent temp-dir so this node is
# recognized as a separate node.
if [ "${1:-}" = "start" ]; then
  mkdir -p "${RAY_TEMP_DIR}"
  ray start --address="${RAY_ADDRESS}" --temp-dir="${RAY_TEMP_DIR}"
  echo "[setup_franka_node.sh] Joined Ray head at ${RAY_ADDRESS}. temp-dir: ${RAY_TEMP_DIR}"
  echo "[setup_franka_node.sh] Verify both nodes: python -c \"import ray; ray.init(address='auto'); print(len([n for n in ray.nodes()]))\""
elif [ "${1:-}" = "stop" ]; then
  # On a single machine, ray stop cleans up all local Ray processes.
  ray stop
fi

unset _SCRIPT_DIR
