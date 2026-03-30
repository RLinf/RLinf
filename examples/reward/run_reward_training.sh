#!/bin/bash
# Copyright 2026 The RLinf Authors.
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

CONFIG_NAME="reward_training"

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${LOG_DIR}"
MEGA_LOG_FILE="${LOG_DIR}/run_reward_training.log"

CMD="$CMD runner.logger.log_path=${LOG_DIR}"

# Run training with remaining arguments
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}

