#!/usr/bin/env bash
set -euo pipefail
ulimit -n 1048576 2>/dev/null || ulimit -n 65536 2>/dev/null || true

ROOT="${RPENT_ROOT:-/path/to/rpent_ppo}"
export PATH=/opt/venv/reason/bin:/usr/local/cuda/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
export PYTHONPATH="${ROOT}/RLinf:${ROOT}/RPent:/opt/venv/reason/Megatron-LM"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

cd "${ROOT}/RLinf"
exec /opt/venv/reason/bin/python examples/agent/rpent/train.py "$@"
