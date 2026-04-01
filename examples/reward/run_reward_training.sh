#!/bin/bash
# Copyright 2026 The RLinf Authors.
# Run Reward Model Training

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

echo "============================================"
echo "Reward Model Training"
echo "============================================"
echo "Project root: $PROJECT_ROOT"

CONFIG_NAME="reward_training"

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}" #/$(date +'%Y%m%d-%H:%M:%S')"
mkdir -p "${LOG_DIR}"
MEGA_LOG_FILE="${LOG_DIR}/run_reward_training.log"

# Build command
CMD="python examples/reward/train_reward_model.py"
CMD="$CMD runner.logger.log_path=${LOG_DIR}"

# Run training with remaining arguments
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}

