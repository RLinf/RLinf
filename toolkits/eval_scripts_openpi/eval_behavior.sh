#!/bin/bash
# BEHAVIOR evaluation script for RLinf models

# Set environment variables
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Default parameters
TASK_NAME="turning_on_radio"
MODEL_PATH="/path/to/model/checkpoint"
POLICY_TYPE="rlinf"
ACTION_CHUNK=32
MAX_STEPS=2000
NUM_EPISODES=1
LOG_PATH="./behavior_eval_results"
NUM_SAVE_VIDEOS=10
SEED=42

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --task_name)
            TASK_NAME="$2"
            shift 2
            ;;
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --policy_type)
            POLICY_TYPE="$2"
            shift 2
            ;;
        --action_chunk)
            ACTION_CHUNK="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --log_path)
            LOG_PATH="$2"
            shift 2
            ;;
        --num_save_videos)
            NUM_SAVE_VIDEOS="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --task_name TASK         BEHAVIOR task name (default: turning_on_radio)"
            echo "  --model_path PATH        Path to model checkpoint (required)"
            echo "  --policy_type TYPE       Policy type: rlinf or openpi (default: rlinf)"
            echo "  --action_chunk N         Action chunk size (default: 32)"
            echo "  --max_steps N            Max steps per episode (default: 2000)"
            echo "  --num_episodes N         Episodes per instance (default: 1)"
            echo "  --log_path PATH          Results directory (default: ./behavior_eval_results)"
            echo "  --num_save_videos N      Number of videos to save (default: 10)"
            echo "  --seed N                 Random seed (default: 42)"
            echo ""
            echo "Example:"
            echo "  $0 --task_name turning_on_radio --model_path /path/to/model --action_chunk 32"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ "$MODEL_PATH" = "/path/to/model/checkpoint" ]; then
    echo "Error: --model_path is required"
    echo "Use --help for usage information"
    exit 1
fi

# Print configuration
echo "========================================"
echo "BEHAVIOR Evaluation Configuration"
echo "========================================"
echo "Task Name:       $TASK_NAME"
echo "Model Path:      $MODEL_PATH"
echo "Policy Type:     $POLICY_TYPE"
echo "Action Chunk:    $ACTION_CHUNK"
echo "Max Steps:       $MAX_STEPS"
echo "Num Episodes:    $NUM_EPISODES"
echo "Log Path:        $LOG_PATH"
echo "Num Videos:      $NUM_SAVE_VIDEOS"
echo "Seed:            $SEED"
echo "========================================"
echo ""

# Run evaluation
python ${REPO_PATH}/toolkits/eval_scripts_openpi/behavior_eval.py \
    --task_name ${TASK_NAME} \
    --pretrained_path ${MODEL_PATH} \
    --policy_type ${POLICY_TYPE} \
    --action_chunk ${ACTION_CHUNK} \
    --max_steps ${MAX_STEPS} \
    --num_episodes_per_instance ${NUM_EPISODES} \
    --log_path ${LOG_PATH} \
    --num_save_videos ${NUM_SAVE_VIDEOS} \
    --seed ${SEED}

echo ""
echo "========================================"
echo "Evaluation completed!"
echo "Results saved to: $LOG_PATH"
echo "========================================"

