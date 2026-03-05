#!/bin/bash
# Compute returns for LeRobot datasets
#
# This script:
# 1. Adds `return`, `reward`, and `prompt` columns to LeRobot datasets
# 2. Updates meta/stats.json with return/reward statistics (mean, std, min, max)
# 3. Updates meta/info.json with new feature definitions
#
# Return computation:
#   - SFT datasets: reward=-1 per step, last step=0, all episodes successful
#   - Rollout datasets: reward=-1 per step, last step=0 (success) or failure_reward (failure)
#   - Returns computed via backward iteration: G_t = r_t + gamma * G_{t+1}
#
# NOTE: Datasets are configured in YAML config file (supports multiple datasets).
#
# Usage:
#   bash run_compute_returns.sh [CONFIG_NAME] [EXTRA_ARGS]
#
# Examples:
#   # Use default config (compute_returns.yaml) - configure datasets in YAML
#   bash run_compute_returns.sh
#
#   # Use custom config
#   bash run_compute_returns.sh my_compute_returns
#
#   # Override data_root
#   bash run_compute_returns.sh compute_returns data.data_root=/path/to/data
#
#   # Single dataset via command line (backward compatible)
#   bash run_compute_returns.sh compute_returns data.dataset_path=/path/to/dataset data.dataset_type=sft

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"
set -e

# Source environment
source switch_env openpi 2>/dev/null || true

# Get script directory and set up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH=$(dirname $(dirname "$SCRIPT_DIR"))
export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH
cd "$SCRIPT_DIR"

# Parse arguments
CONFIG_NAME=${1:-"compute_returns"}  # Default config
shift 1 2>/dev/null || true
EXTRA_ARGS="$@"

# Build config overrides
OVERRIDES=""
if [ -n "$EXTRA_ARGS" ]; then
    OVERRIDES="$EXTRA_ARGS"
fi

echo "=========================================="
echo "Return Computation for LeRobot Datasets"
echo "=========================================="
echo "Configuration:"
echo "  - Config: $CONFIG_NAME"
if [ -n "$EXTRA_ARGS" ]; then
    echo "  - Extra args: $EXTRA_ARGS"
fi
echo ""
echo "Output (for each dataset):"
echo "  - Adds 'return', 'reward', 'prompt' columns to parquet files"
echo "  - Updates meta/stats.json with statistics"
echo "  - Updates meta/info.json with feature definitions"
echo ""
echo "Note: Configure datasets in YAML file (data.datasets list)"
echo ""

# Build command
CMD="python compute_returns.py --config-name $CONFIG_NAME $OVERRIDES"

echo "Command: $CMD"
echo ""

# Run
eval $CMD

echo ""
echo "=========================================="
echo "Return computation complete!"
echo "=========================================="
