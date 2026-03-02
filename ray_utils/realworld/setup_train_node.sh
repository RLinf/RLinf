#!/bin/bash
# ============================================================
# 训练容器 (maniskill_libero) 专用 — Ray Head 节点, rank 0
# 在训练容器中 source 此脚本后再启动 Ray
# 用法: source ray_utils/realworld/setup_train_node.sh
# ============================================================

export CURRENT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$CURRENT_PATH"))
export PYTHONPATH=$REPO_PATH:$PYTHONPATH

# ---- RLinf 节点配置 ----
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES="enp5s0"  # 用于跨容器通信的网卡（两个容器共享宿主机网络）

# ---- 修复 CUDA compat 库问题（535 驱动 + 12.4 compat 在消费级 GPU 上报 Error 804）----
if [ -f /etc/ld.so.conf.d/00-compat-70194656.conf ]; then
    rm -f /etc/ld.so.conf.d/00-compat-70194656.conf
    echo "/usr/local/cuda-12.1/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/000_cuda.conf
    ln -sf /usr/local/cuda-12.1 /usr/local/cuda
    ldconfig 2>/dev/null
    echo "[INFO] 已移除 CUDA compat 库，切换到 CUDA 12.1"
fi
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

# ---- 激活训练用的 Python 虚拟环境 ----
source switch_env openvla

echo "===== 训练节点环境已就绪 ====="
echo "  RLINF_NODE_RANK  = $RLINF_NODE_RANK"
echo "  RLINF_COMM_NET_DEVICES = $RLINF_COMM_NET_DEVICES"
echo "  REPO_PATH        = $REPO_PATH"
echo "  Python            = $(which python)"
echo "================================"
