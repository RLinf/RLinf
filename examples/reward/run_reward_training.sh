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
DATA_PATH=""

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
        --data)
            DATA_PATH="$2"
            shift 2
            ;;
        *)
            # If first positional arg and DATA_PATH not set, treat as data path
            if [[ -z "$DATA_PATH" && ! "$1" =~ ^-- && ! "$1" =~ = ]]; then
                DATA_PATH="$1"
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

# Add data path if provided
if [[ -n "$DATA_PATH" ]]; then
    # Convert relative path to absolute if needed
    if [[ ! "$DATA_PATH" =~ ^/ ]]; then
        DATA_PATH="$PROJECT_ROOT/$DATA_PATH"
    fi
    echo "Data path: $DATA_PATH"
    CMD="$CMD data.data_path=$DATA_PATH"
else
    echo "Data path: (using config default)"
fi
echo "============================================"

# Run training with remaining arguments
$CMD "$@"

