#!/bin/bash
# ============================================================
# Training container (maniskill_libero) — Ray Head node, rank 0
# Source this script inside the training container before starting Ray.
# Usage: source ray_utils/realworld/setup_train_node.sh
# ============================================================

export CURRENT_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$CURRENT_PATH"))
export PYTHONPATH=$REPO_PATH:$PYTHONPATH

# ---- RLinf node config ----
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES="enp5s0"  # NIC for cross-container communication (both containers share the host network)

# ---- Fix CUDA compat library issue (535 driver + 12.4 compat causes Error 804 on consumer GPUs) ----
if [ -f /etc/ld.so.conf.d/00-compat-70194656.conf ]; then
    rm -f /etc/ld.so.conf.d/00-compat-70194656.conf
    echo "/usr/local/cuda-12.1/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/000_cuda.conf
    ln -sf /usr/local/cuda-12.1 /usr/local/cuda
    ldconfig 2>/dev/null
    echo "[INFO] Removed CUDA compat library, switched to CUDA 12.1"
fi
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:${LD_LIBRARY_PATH:-}

# ---- Activate training Python virtual environment ----
source switch_env openvla

echo "===== Training node environment ready ====="
echo "  RLINF_NODE_RANK  = $RLINF_NODE_RANK"
echo "  RLINF_COMM_NET_DEVICES = $RLINF_COMM_NET_DEVICES"
echo "  REPO_PATH        = $REPO_PATH"
echo "  Python            = $(which python)"
echo "================================"
