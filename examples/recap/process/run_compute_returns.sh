#!/bin/bash
# Compute returns for LeRobot datasets
#
# Usage:
#   bash run_compute_returns.sh [CONFIG_NAME] [EXTRA_ARGS]
#
# Examples:
#   bash run_compute_returns.sh
#   bash run_compute_returns.sh compute_returns data.data_root=/path/to/data
#   bash run_compute_returns.sh compute_returns data.dataset_path=/path/to/dataset data.dataset_type=sft

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"
set -e

source switch_env openpi 2>/dev/null || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))
export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH
cd "$SCRIPT_DIR"

CONFIG_NAME=${1:-"compute_returns"}
shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

OVERRIDES=""
if [ -n "$EXTRA_ARGS" ]; then
    OVERRIDES="$EXTRA_ARGS"
fi

echo "=========================================="
echo "Return Computation for LeRobot Datasets"
echo "=========================================="
echo "  Config: $CONFIG_NAME"
if [ -n "$EXTRA_ARGS" ]; then
    echo "  Extra args: $EXTRA_ARGS"
fi
echo ""

CMD="python compute_returns.py --config-name $CONFIG_NAME $OVERRIDES"

echo "Command: $CMD"
echo ""

eval $CMD

echo ""
echo "=========================================="
echo "Return computation complete!"
echo "=========================================="
