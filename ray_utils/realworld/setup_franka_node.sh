#!/bin/bash
# ============================================================
# Franka control container — Ray Worker node, rank 1
# Source this script inside the Franka container before starting Ray.
# Usage: source ray_utils/realworld/setup_franka_node.sh
# ============================================================

export CURRENT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$CURRENT_PATH"))
export PYTHONPATH="/opt/venv/franka-0.15.0/franka_catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:/workspace/RLinf"
export LD_LIBRARY_PATH="/opt/venv/franka-0.15.0/franka_catkin_ws/devel/lib:/opt/venv/franka-0.15.0/franka_catkin_ws/libfranka/build:/opt/ros/noetic/lib:/opt/openrobots/lib"

# ---- RLinf node config ----
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES="enp5s0"  # NIC for cross-container communication (both containers share the host network)

# ---- Activate Franka Python virtual environment ----
# Choose the libfranka version matching your Franka firmware.
# Options: franka-0.10.0 / franka-0.13.3 / franka-0.14.1 / franka-0.15.0 / franka-0.18.0
source switch_env franka-0.15.0

echo "===== Franka control node environment ready ====="
echo "  RLINF_NODE_RANK  = $RLINF_NODE_RANK"
echo "  RLINF_COMM_NET_DEVICES = $RLINF_COMM_NET_DEVICES"
echo "  REPO_PATH        = $REPO_PATH"
echo "  Python            = $(which python)"
echo "======================================="
