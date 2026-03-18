#!/bin/bash
# Copyright 2025 The RLinf Authors.
# Run ResNet Reward Model Training

set -e

# Disable torch dynamo to avoid jinja2 compatibility issues
export TORCHDYNAMO_DISABLE=1

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Parse arguments
EARLY_STOP="true"
PATIENCE=5
TRAIN_DATA_PATH=""
VAL_DATA_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-early-stop)
            EARLY_STOP="false"
            shift
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --train-data)
            TRAIN_DATA_PATH="$2"
            shift 2
            ;;
        --val-data)
            VAL_DATA_PATH="$2"
            shift 2
            ;;
        *)
            # If first positional arg and TRAIN_DATA_PATH not set, treat as train data path
            if [[ -z "$TRAIN_DATA_PATH" && ! "$1" =~ ^-- && ! "$1" =~ = ]]; then
                TRAIN_DATA_PATH="$1"
                shift
            else
                break
            fi
            ;;
    esac
done

echo "============================================"
echo "ResNet Reward Model Training"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Early stop: $EARLY_STOP (patience: $PATIENCE)"

# Build command
CMD="python examples/reward/train_reward_model.py"
CMD="$CMD runner.early_stop.enabled=$EARLY_STOP"
CMD="$CMD runner.early_stop.patience=$PATIENCE"

# Add train data path if provided
if [[ -n "$TRAIN_DATA_PATH" ]]; then
    # Convert relative path to absolute if needed
    if [[ ! "$TRAIN_DATA_PATH" =~ ^/ ]]; then
        TRAIN_DATA_PATH="$PROJECT_ROOT/$TRAIN_DATA_PATH"
    fi
    echo "Train data path: $TRAIN_DATA_PATH"
    CMD="$CMD data.train_data_path=$TRAIN_DATA_PATH"
else
    echo "Train data path: (using config default)"
fi

# Add val data path if provided
if [[ -n "$VAL_DATA_PATH" ]]; then
    if [[ ! "$VAL_DATA_PATH" =~ ^/ ]]; then
        VAL_DATA_PATH="$PROJECT_ROOT/$VAL_DATA_PATH"
    fi
    echo "Val data path: $VAL_DATA_PATH"
    CMD="$CMD data.val_data_path=$VAL_DATA_PATH"
else
    echo "Val data path: (using config default)"
fi
echo "============================================"

# Run training with remaining arguments
$CMD "$@"

