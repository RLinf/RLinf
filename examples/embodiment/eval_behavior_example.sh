#!/bin/bash
# Example script for evaluating BEHAVIOR models using RLinf's worker-based architecture
#
# This demonstrates the recommended approach for BEHAVIOR evaluation:
# - Uses RLinf's parallel worker architecture (env workers + rollout workers)
# - Faster than single-threaded evaluation
# - Follows RLinf's standard evaluation pattern
#
# For detailed per-instance metrics with single-threaded evaluation, use:
#   toolkits/eval_scripts_openpi/behavior_eval.py

set -e

# Configuration
TASK_NAME="turning_on_radio"  # Task name from BEHAVIOR-1K
# MODEL_PATH="/mnt/public/quanlu/pi05-b1kpt12-cs32"  # Path to your trained model
MODEL_PATH="/mnt/public/quanlu/RLinf-OpenVLAOFT-Behavior/"
EVAL_CONFIG="behavior_openvlaoft_eval"  # Evaluation config file
LOG_PATH="./logs/behavior_eval"  # Output directory

# ============================================================================
# Method 1: Parallel Evaluation with RLinf Workers (Recommended)
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
echo "========================================" echo "Method 1: RLinf Parallel Evaluation"
echo "========================================"

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
echo ""

# ============================================================================
# Method 2: Single-threaded Evaluation with Detailed Metrics (Optional)
# ============================================================================
# This method provides more granular metrics but is slower.
# Use this if you need:
# - Detailed per-instance statistics
# - Step-by-step debugging
# - Custom evaluation logic
echo "========================================"
echo "Method 2: Single-threaded Evaluation"
echo "========================================"
echo "(Commented out by default - uncomment to use)"
echo ""

# Uncomment to run single-threaded evaluation:
# bash ${REPO_PATH}/toolkits/eval_scripts_openpi/eval_behavior.sh \
#     --task_name ${TASK_NAME} \
#     --model_path ${MODEL_PATH} \
#     --policy_type rlinf \
#     --action_chunk 32 \
#     --max_steps 2000 \
#     --num_episodes 1 \
#     --log_path ${LOG_PATH}/single_threaded \
#     --num_save_videos 10

# ============================================================================
# Evaluation Configuration Options
# ============================================================================
# Key parameters you can adjust:
#
# Task Configuration:
#   env.eval.task_idx: Task index (0-49 for BEHAVIOR-1K tasks)
#   env.eval.total_num_envs: Number of parallel environments
#
# Episode Configuration:
#   env.eval.max_steps_per_rollout_epoch: Max steps per episode
#   algorithm.eval_rollout_epoch: Number of evaluation epochs
#
# Model Configuration:
#   actor.model.num_action_chunks: Action chunk size (must match training)
#   runner.eval_policy_path: Path to model checkpoint
#
# Video Configuration:
#   env.eval.video_cfg.save_video: Whether to save videos
#   env.eval.video_cfg.video_base_dir: Video output directory
#
# For more options, see: examples/embodiment/config/behavior_openvlaoft_eval.yaml

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

# ============================================================================
# Multi-Task Evaluation Example
# ============================================================================
# To evaluate on multiple tasks, run the script multiple times with different
# task indices:

# for task_idx in 0 1 2 3 4; do
#     echo "Evaluating task index: ${task_idx}"
#     python ${EMBODIED_PATH}/eval_embodied_agent.py \
#         --config-name ${EVAL_CONFIG} \
#         runner.logger.log_path=${LOG_PATH}/task_${task_idx} \
#         runner.eval_policy_path=${MODEL_PATH}/model.pt \
#         env.eval.task_idx=${task_idx} \
#         algorithm.eval_rollout_epoch=10
# done

echo "========================================" echo "All evaluations complete!"
echo "========================================"

