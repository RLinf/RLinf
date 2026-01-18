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

DATA_PATH="$1"
shift  # Remove first argument from $@

# Convert relative path to absolute if needed
if [[ ! "$DATA_PATH" =~ ^/ ]]; then
    DATA_PATH="$PROJECT_ROOT/$DATA_PATH"
fi

echo "============================================"
echo "ResNet Reward Model Training"
echo "============================================"
echo "Project root: $PROJECT_ROOT"
echo "Data path: $DATA_PATH"
echo "============================================"

# Run training
python examples/reward/train_reward_model.py \
    data.data_path="$DATA_PATH" \
    "$@"

