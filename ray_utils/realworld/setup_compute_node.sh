#!/bin/bash
# setup_compute_node.sh - Startup script for the compute/rollout node (Ray head).
# Usage:
#   source setup_compute_node.sh          # Activate the compute environment
#   source setup_compute_node.sh start    # Activate env and start the Ray head (rank 0)
#   source setup_compute_node.sh stop     # Stop Ray on this node
#
# This node only serves the compute/rollout roles (actor / rollout / reward).
# The control node (Franka) is started via setup_franka_node.sh and joins this
# head as rank 1.
#
# Required environment variables:
#   RLINF_NODE_IP  Reachable IP address for this Ray head (required with `start`).
# Optional environment variables:
#   RLINF_VENV, RLINF_COMM_NET_DEVICES, RAY_TEMP_DIR

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_PATH="$(dirname "$(dirname "$_SCRIPT_DIR")")"

# Compute environment. Defaults to the repository's .venv.
export RLINF_VENV="${RLINF_VENV:-${REPO_PATH}/.venv}"
# The compute node is fixed as rank 0 (head).
export RLINF_NODE_RANK=0
# Set RLINF_COMM_NET_DEVICES to the network interface that other nodes can reach.
export RLINF_COMM_NET_DEVICES="${RLINF_COMM_NET_DEVICES:-eth0}"

# Ray uses an independent temp dir so that the control node (with a different
# --temp-dir) is recognized as a separate node. Use a short path under /tmp to
# stay within the 107-byte AF_UNIX socket length limit.
export RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/rlinf_compute}"

if [ "${1:-}" = "start" ] && [ -z "${RLINF_NODE_IP:-}" ]; then
  echo "[setup_compute_node.sh] ERROR: RLINF_NODE_IP is required with 'start'." >&2
  echo "  export RLINF_NODE_IP=<this_node_reachable_ip>" >&2
  unset _SCRIPT_DIR
  return 1 2>/dev/null || exit 1
fi
if [ -n "${RLINF_NODE_IP:-}" ]; then
  export RLINF_NODE_IP
fi

# Avoid interference from conda / PYTHONHOME.
if [ -n "${CONDA_PREFIX:-}" ] && type conda >/dev/null 2>&1; then
  for _i in 1 2 3; do
    [ -z "${CONDA_PREFIX:-}" ] && break
    conda deactivate || break
  done
fi
unset PYTHONHOME
unset PYTHONPATH

if [ ! -f "${RLINF_VENV}/bin/activate" ]; then
  echo "[setup_compute_node.sh] ERROR: venv not found: ${RLINF_VENV}/bin/activate" >&2
  echo "  cd ${REPO_PATH} && bash requirements/install.sh embodied --model <model> --env <env> --venv ${RLINF_VENV}" >&2
  unset _SCRIPT_DIR
  return 1 2>/dev/null || exit 1
fi

# The control node sources ROS / catkin; the compute node does not need it.
source "${RLINF_VENV}/bin/activate"
hash -r

# Put this repo first on PYTHONPATH.
export PYTHONPATH="${REPO_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[setup_compute_node.sh] role: compute/rollout"
echo "[setup_compute_node.sh] repo: ${REPO_PATH}"
echo "[setup_compute_node.sh] venv: ${RLINF_VENV}"
echo "[setup_compute_node.sh] node_rank: ${RLINF_NODE_RANK}"
echo "[setup_compute_node.sh] python: $(which python)"

python -c "import rlinf; print('[setup_compute_node.sh] rlinf:', rlinf.__file__)" || {
  echo "[setup_compute_node.sh] ERROR: cannot import rlinf. Try:" >&2
  echo "  cd ${REPO_PATH} && pip install -e ." >&2
  unset _SCRIPT_DIR
  return 1 2>/dev/null || exit 1
}

# Start the Ray head (compute node, rank 0). The control node must join with a
# different --temp-dir so it is recognized as a separate node.
if [ "${1:-}" = "start" ]; then
  mkdir -p "${RAY_TEMP_DIR}"
  ray start --head --port=6379 --node-ip-address="${RLINF_NODE_IP}" \
    --temp-dir="${RAY_TEMP_DIR}"
  echo "[setup_compute_node.sh] Ray head started. temp-dir: ${RAY_TEMP_DIR}  ip: ${RLINF_NODE_IP}"
  echo "[setup_compute_node.sh] Start the control node with: source <repo>/ray_utils/realworld/setup_franka_node.sh start"
elif [ "${1:-}" = "stop" ]; then
  # On a single machine, ray stop cleans up all local Ray processes.
  ray stop
fi

unset _SCRIPT_DIR
