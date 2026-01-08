#!/bin/bash
# Script to run the resnet/RLinf project inside the rlinf Docker container
# This script uses the existing rlinf container's environment but with
# the current project's codebase.

set -e

# Project paths
RESNET_RLINF_PATH="/mnt/mnt/public_zgc/zhoutianxing/resnet/RLinf"
ORIGINAL_RLINF_PATH="/mnt/mnt/public_zgc/zhoutianxing/RLinf"
CONTAINER_NAME="rlinf"

# Check if container is running
if ! docker ps | grep -q "$CONTAINER_NAME"; then
    echo "Error: Container '$CONTAINER_NAME' is not running."
    echo "Please start it first."
    exit 1
fi

# Function to show usage
usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  shell       - Open interactive shell in container (default)"
    echo "  train       - Run training with config (requires CONFIG_NAME env var)"
    echo "  test        - Test reward model import"
    echo "  <custom>    - Run any custom command"
    echo ""
    echo "Examples:"
    echo "  $0 shell"
    echo "  CONFIG_NAME=maniskill_sac_mlp $0 train"
    echo "  $0 python -c 'from rlinf.algorithms.rewards.embodiment import RewardManager; print(\"OK\")'"
}

# Setup command to run inside container
# Sets PYTHONPATH to use resnet/RLinf instead of original RLinf
SETUP_CMD="export PYTHONPATH=${RESNET_RLINF_PATH}:\$PYTHONPATH && cd ${RESNET_RLINF_PATH}"

case "${1:-shell}" in
    shell)
        echo "Entering container with resnet/RLinf environment..."
        docker exec -it "$CONTAINER_NAME" bash -c "$SETUP_CMD && bash"
        ;;
    train)
        if [ -z "$CONFIG_NAME" ]; then
            echo "Error: CONFIG_NAME environment variable is required for training."
            echo "Example: CONFIG_NAME=maniskill_sac_mlp $0 train"
            exit 1
        fi
        echo "Running training with config: $CONFIG_NAME"
        docker exec -it "$CONTAINER_NAME" bash -c "$SETUP_CMD && cd examples/embodiment && ./run_embodiment.sh $CONFIG_NAME"
        ;;
    test)
        echo "Testing reward model import..."
        docker exec -it "$CONTAINER_NAME" bash -c "$SETUP_CMD && python -c '
from rlinf.algorithms.rewards.embodiment import (
    RewardManager,
    BaseRewardModel,
    BaseImageRewardModel,
    BaseVideoRewardModel,
    ResNetRewardModel,
    Qwen3VLRewardModel,
)
print(\"✓ All imports successful!\")
print(f\"  Available models: {RewardManager.get_available_models()}\")
'"
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        # Run custom command
        docker exec -it "$CONTAINER_NAME" bash -c "$SETUP_CMD && $*"
        ;;
esac

