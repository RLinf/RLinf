#!/bin/bash
# Compute advantages: A = normalize(r_{t:t+N}) + gamma^N * V(o_{t+N}) - V(o_t)
#
# Usage:
#   bash run_compute_advantages.sh CONFIG_NAME [--nproc N] [HYDRA_OVERRIDES...]
#
# Examples:
#   # Default config, all available GPUs
#   bash run_compute_advantages.sh compute_advantages
#
#   # Specify GPU count
#   bash run_compute_advantages.sh compute_advantages --nproc 4
#
#   # Custom config with 8 GPUs
#   bash run_compute_advantages.sh compute_advantages_libero_3shot_collect_4096_thresh15 --nproc 8
#
#   # With Hydra overrides
#   bash run_compute_advantages.sh compute_advantages --nproc 4 advantage.tag=model_a

set -e

source switch_env openpi 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))
export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH
cd "$SCRIPT_DIR"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-/tmp}"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TMPDIR"

CONFIG_NAME="${1:-compute_advantages}"
shift 1 2>/dev/null || true

NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)
OVERRIDES=""

while [ $# -gt 0 ]; do
    case "$1" in
        --nproc)
            NPROC_PER_NODE="$2"
            shift 2
            ;;
        *)
            OVERRIDES="$OVERRIDES $1"
            shift
            ;;
    esac
done

if [ "$NPROC_PER_NODE" -lt 1 ]; then
    NPROC_PER_NODE=1
fi

MASTER_PORT=${MASTER_PORT:-29500}
while netstat -tuln 2>/dev/null | grep -q ":$MASTER_PORT " || ss -tuln 2>/dev/null | grep -q ":$MASTER_PORT "; do
    MASTER_PORT=$((MASTER_PORT + 1))
    if [ $MASTER_PORT -gt 30000 ]; then
        echo "Warning: Could not find available port, using default 29500"
        MASTER_PORT=29500
        break
    fi
done

echo "=========================================="
echo "Advantage Computation"
echo "=========================================="
echo "  GPUs: $NPROC_PER_NODE"
echo "  Config: $CONFIG_NAME"
echo "  Master port: $MASTER_PORT"
if [ -n "$OVERRIDES" ]; then
    echo "  Overrides: $OVERRIDES"
fi
echo ""

if [ "$NPROC_PER_NODE" -eq 1 ]; then
    CMD="python compute_advantages.py --config-name $CONFIG_NAME $OVERRIDES"
    echo "Running single-GPU mode..."
else
    CMD="torchrun \
        --nproc_per_node=$NPROC_PER_NODE \
        --master_port=$MASTER_PORT \
        compute_advantages.py \
        --config-name $CONFIG_NAME \
        $OVERRIDES"
    echo "Running multi-GPU mode with torchrun..."
fi

echo ""
echo "Command: $CMD"
echo ""

eval $CMD 2> >(grep -v "libdav1d" >&2)

echo ""
echo "=========================================="
echo "Advantage computation complete!"
echo "=========================================="
