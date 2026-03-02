#!/bin/bash
# ============================================================
# Franka 控制容器专用 — Ray Worker 节点, rank 1
# 在 Franka 容器中 source 此脚本后再启动 Ray
# 用法: source ray_utils/realworld/setup_franka_node.sh
# ============================================================

export CURRENT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$CURRENT_PATH"))
export PYTHONPATH="/opt/venv/franka-0.15.0/franka_catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:/workspace/RLinf"
export LD_LIBRARY_PATH="/opt/venv/franka-0.15.0/franka_catkin_ws/devel/lib:/opt/venv/franka-0.15.0/franka_catkin_ws/libfranka/build:/opt/ros/noetic/lib:/opt/openrobots/lib"

# ---- RLinf 节点配置 ----
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES="enp5s0"  # 用于跨容器通信的网卡（两个容器共享宿主机网络）

# ---- 激活 Franka 控制用的 Python 虚拟环境 ----
# 根据你的 Franka 固件版本选择对应的 libfranka 版本
# 可选: franka-0.10.0 / franka-0.13.3 / franka-0.14.1 / franka-0.15.0 / franka-0.18.0
source switch_env franka-0.15.0

echo "===== Franka 控制节点环境已就绪 ====="
echo "  RLINF_NODE_RANK  = $RLINF_NODE_RANK"
echo "  RLINF_COMM_NET_DEVICES = $RLINF_COMM_NET_DEVICES"
echo "  REPO_PATH        = $REPO_PATH"
echo "  Python            = $(which python)"
echo "======================================="
