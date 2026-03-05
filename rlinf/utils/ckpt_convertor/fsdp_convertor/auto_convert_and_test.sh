#!/bin/bash
set -e
set -x
set -o pipefail

# Auto Checkpoint Conversion and Testing Script
# This script automates:
# 1. Converting FSDP checkpoint to HuggingFace format
# 2. Running a series of evaluation tests with the converted checkpoint

# Input: checkpoint path (e.g., .../model_state_dict/)
#        eval name (used as the log folder name)
CKPT_DIR=$1
EVAL_NAME=$2

if [ -z "$CKPT_DIR" ] || [ -z "$EVAL_NAME" ]; then
    echo "Usage: $0 <checkpoint_directory_path> <eval_name>"
    echo "Example: $0 /path/to/logs/checkpoints/global_step_1000/actor/model_state_dict/ step1000_eval"
    echo ""
    echo "  <eval_name>: A name for this evaluation run. All logs will be saved under:"
    echo "               \${RELEASE_DIR}/logs/<eval_name>/"
    exit 1
fi

# Verify the checkpoint directory exists
if [ ! -d "$CKPT_DIR" ]; then
    echo "Error: Checkpoint directory does not exist: $CKPT_DIR"
    exit 1
fi

# Generate unique ID for temporary files (timestamp + PID)
UNIQUE_ID=$(date +%Y%m%d_%H%M%S)_$$

# === Step 1: Checkpoint Conversion ===
echo "=== Step 1: Checkpoint Conversion ==="

SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname $(dirname "$SCRIPT_DIR")))

# Check if safetensors already exist (skip conversion if so)
SAFETENSOR_FILES=$(find "$CKPT_DIR" -maxdepth 1 -name "*.safetensors" 2>/dev/null)
if [ -n "$SAFETENSOR_FILES" ]; then
    echo "Found existing safetensors in ${CKPT_DIR}, skipping conversion."
else
    # Create temporary convertor config
    TEMP_CONFIG="${SCRIPT_DIR}/config/temp_convertor_${UNIQUE_ID}.yaml"
    cp "${SCRIPT_DIR}/config/fsdp_model_convertor.yaml" "$TEMP_CONFIG"

    # Update save_path and ckpt_path in temporary config using sed
    # Use | as delimiter since paths contain /
    sed -i "s|save_path:.*|save_path: ${CKPT_DIR}|g" "$TEMP_CONFIG"
    sed -i "s|ckpt_path:.*|ckpt_path: ${CKPT_DIR}/full_weights.pt|g" "$TEMP_CONFIG"

    echo "Created temporary config: $TEMP_CONFIG"
    echo "Converting checkpoint from: ${CKPT_DIR}/full_weights.pt"
    echo "Output path: ${CKPT_DIR}"

    # Run conversion
    export REPO_PATH=${REPO_PATH}
    export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
    python ${SCRIPT_DIR}/convert_pt_to_hf.py \
        --config-path ${SCRIPT_DIR}/config \
        --config-name temp_convertor_${UNIQUE_ID}

    CONVERT_EXIT_CODE=$?

    # Clean up temp convertor config
    rm -f "$TEMP_CONFIG"

    if [ $CONVERT_EXIT_CODE -ne 0 ]; then
        echo "Error: Checkpoint conversion failed with exit code $CONVERT_EXIT_CODE"
        exit $CONVERT_EXIT_CODE
    fi

    echo "Checkpoint conversion completed successfully!"
fi

# === Step 2: Environment Setup ===
echo "=== Step 2: Environment Setup ==="

RELEASE_DIR="${RELEASE_DIR:-/path/to/RLinf_release}"
cd "$RELEASE_DIR"
source switch_env openpi

export PYTHONPATH=${RELEASE_DIR}:$PYTHONPATH
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export EMBODIED_PATH="${RELEASE_DIR}/examples/embodiment"

# === Step 3: Run Tests ===
echo "=== Step 3: Run Evaluation Tests ==="

EVAL_CONFIG_DIR="${RELEASE_DIR}/examples/embodiment/config"
SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

# List of test configs
CONFIGS=(
    "libero_10_pi06_eval_positive_10"
    "libero_10_pi06_eval_no_guide"
    "libero_10_pi06_eval_positive_20"
    "libero_10_pi06_eval_positive_40"
    "libero_10_pi06_eval_negative_10"
    "libero_10_pi06_eval_negative_20"
    "libero_10_pi06_eval_negative_40"
)

EVAL_LOG_ROOT="${RELEASE_DIR}/logs/${EVAL_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${EVAL_LOG_ROOT}"

FAILED_TESTS=()
PASSED_TESTS=()

for CONFIG_NAME in "${CONFIGS[@]}"; do
    echo ""
    echo "========================================"
    echo "Running test: ${CONFIG_NAME}"
    echo "========================================"

    ORIGINAL_CONFIG="${EVAL_CONFIG_DIR}/${CONFIG_NAME}.yaml"

    # Check if the original config exists
    if [ ! -f "$ORIGINAL_CONFIG" ]; then
        echo "Warning: Config file does not exist: $ORIGINAL_CONFIG"
        echo "Skipping test: ${CONFIG_NAME}"
        FAILED_TESTS+=("${CONFIG_NAME} (config not found)")
        continue
    fi

    # Create temporary eval config with unique name
    TEMP_CONFIG_NAME="${CONFIG_NAME}_temp_${UNIQUE_ID}"
    TEMP_EVAL_CONFIG="${EVAL_CONFIG_DIR}/${TEMP_CONFIG_NAME}.yaml"

    # Copy original and replace model_path entries
    # Replace both rollout.model.model_path and actor.model.model_path
    # The pattern matches paths that contain 'model_state_dict' or full /mnt/ paths
    # Remove trailing slash from CKPT_DIR if present for consistency
    CKPT_DIR_CLEAN="${CKPT_DIR%/}"

    # Use sed to replace model_path lines that contain checkpoint paths
    sed "s|model_path: \"/mnt/[^\"]*\"|model_path: \"${CKPT_DIR_CLEAN}\"|g" \
        "$ORIGINAL_CONFIG" > "$TEMP_EVAL_CONFIG"

    echo "Created temporary config: $TEMP_EVAL_CONFIG"

    # Run evaluation
    LOG_DIR="${EVAL_LOG_ROOT}/${CONFIG_NAME}"
    MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
    mkdir -p "${LOG_DIR}"

    CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${TEMP_CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
    echo "Running: ${CMD}"
    ${CMD} 2>&1 | tee ${MEGA_LOG_FILE}

    TEST_EXIT_CODE=$?

    # Clean up temporary config
    rm -f "$TEMP_EVAL_CONFIG"

    if [ $TEST_EXIT_CODE -ne 0 ]; then
        echo "Warning: Test ${CONFIG_NAME} failed with exit code $TEST_EXIT_CODE"
        FAILED_TESTS+=("${CONFIG_NAME}")
    else
        echo "Test ${CONFIG_NAME} completed successfully!"
        PASSED_TESTS+=("${CONFIG_NAME}")
    fi
done

# === Summary ===
echo ""
echo "========================================"
echo "All Tests Summary"
echo "========================================"
echo "Checkpoint: ${CKPT_DIR}"
echo "Logs directory: ${EVAL_LOG_ROOT}/"
echo ""
echo "Passed tests (${#PASSED_TESTS[@]}):"
for test in "${PASSED_TESTS[@]}"; do
    echo "  - $test"
done
echo ""
echo "Failed tests (${#FAILED_TESTS[@]}):"
for test in "${FAILED_TESTS[@]}"; do
    echo "  - $test"
done
echo ""

if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "All tests completed successfully!"
    exit 0
else
    echo "Some tests failed. Please check the logs for details."
    exit 1
fi
