#!/bin/bash
# Example script for evaluating BEHAVIOR models using RLinf's worker-based architecture
#
# This demonstrates the recommended approach for BEHAVIOR evaluation:
# - Uses RLinf's parallel worker architecture (env workers + rollout workers)
# - Faster than single-threaded evaluation
# - Follows RLinf's standard evaluation pattern
#
# For complete documentation, see:
#   examples/embodiment/BEHAVIOR_EVALUATION.md

set -e

# Configuration
TASK_NAME="turning_on_radio"  # Task name from BEHAVIOR-1K
# MODEL_PATH="/mnt/public/quanlu/pi05-b1kpt12-cs32"  # Path to your trained model
MODEL_PATH="/mnt/public/quanlu/RLinf-OpenVLAOFT-Behavior/"
EVAL_CONFIG="behavior_openvlaoft_eval"  # Evaluation config file
LOG_PATH="./logs/behavior_eval"  # Output directory

# ============================================================================
# Parallel Evaluation with RLinf Workers
# ============================================================================
# This method uses RLinf's distributed worker architecture with:
# - Environment workers: Run OmniGibson BEHAVIOR environments
# - Rollout workers: Run the policy model
# - Ray channels: Communication between workers
# 
# Advantages:
# - Much faster (parallel execution)
# - Scales to multiple GPUs/nodes
# - Follows RLinf's standard architecture
echo "RLinf Parallel Evaluation Starting..."

# Set environment variables
export EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH=$(dirname "$EMBODIED_PATH")
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Run evaluation using RLinf's embodied eval runner
# Note: Most configurations are in the YAML file. Override only what's needed here.
python ${EMBODIED_PATH}/eval_embodied_agent.py \
    --config-name ${EVAL_CONFIG} \
    runner.logger.log_path=${LOG_PATH} \
    rollout.model.model_path=${MODEL_PATH} \
    env.eval.task_idx=0

echo ""
echo "Evaluation completed! Results saved to: ${LOG_PATH}"

# ============================================================================
# BEHAVIOR Task List
# ============================================================================
# The BEHAVIOR benchmark includes 50 household tasks. Common examples:
#
# Task Index | Task Name
# -----------|---------------------------------------------
#     0      | turning_on_radio
#     1      | opening_packages
#     2      | packing_grocery_items_into_bags
#     3      | cleaning_windows
#     4      | setting_table
#     ...    | ...
#
# For the full list, see:
#   rlinf/envs/behavior/behavior_task.jsonl
#   or OmniGibson's TASK_INDICES_TO_NAMES

echo "=========================================="
echo "For more information and troubleshooting:"
echo "  examples/embodiment/BEHAVIOR_EVALUATION.md"
echo "=========================================="

