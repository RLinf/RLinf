#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_REPO_PATH="$(dirname "$SCRIPT_DIR")"

# This wrapper lives in <RLinf root>/rl_test.
# Keep the main yaml in rl_test, but always use RLinf's embodied trainer/config groups.
export REPO_PATH="${REPO_PATH:-$DEFAULT_REPO_PATH}"
export EMBODIED_PATH="${RLINF_EMBODIED_PATH:-${REPO_PATH}/examples/embodiment}"
export CONFIG_DIR="${RLINF_CONFIG_DIR:-$SCRIPT_DIR}"
export EMBODIED_CONFIG_DIR="${EMBODIED_CONFIG_DIR:-${EMBODIED_PATH}/config}"
export SRC_FILE="${SRC_FILE:-${EMBODIED_PATH}/train_embodied_agent.py}"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export PYTHONPATH="${REPO_PATH}:${ROBOTWIN_PATH:-}:${PYTHONPATH:-}"

if [ -z "${1:-}" ]; then
    CONFIG_NAME="${RLINF_CONFIG_NAME:-maniskill_ppo_openvla_quickstart}"
else
    CONFIG_NAME="$1"
fi

ROBOT_PLATFORM="${2:-${ROBOT_PLATFORM:-LIBERO}}"
export ROBOT_PLATFORM

export LIBERO_TYPE="${LIBERO_TYPE:-standard}"
if [ "$LIBERO_TYPE" = "pro" ]; then
    export LIBERO_PERTURBATION="all"
    echo "Evaluation Mode: LIBERO-PRO | Perturbation: $LIBERO_PERTURBATION"
elif [ "$LIBERO_TYPE" = "plus" ]; then
    export LIBERO_SUFFIX="all"
    echo "Evaluation Mode: LIBERO-PLUS | Suffix: $LIBERO_SUFFIX"
else
    echo "Evaluation Mode: Standard LIBERO"
fi

if [ ! -f "$SRC_FILE" ]; then
    echo "train_embodied_agent.py not found: $SRC_FILE" >&2
    exit 1
fi

if [ ! -d "$EMBODIED_CONFIG_DIR" ]; then
    echo "embodied config dir not found: $EMBODIED_CONFIG_DIR" >&2
    exit 1
fi

if [ ! -f "${CONFIG_DIR}/${CONFIG_NAME}.yaml" ]; then
    echo "config yaml not found: ${CONFIG_DIR}/${CONFIG_NAME}.yaml" >&2
    exit 1
fi

echo "Using ROBOT_PLATFORM=$ROBOT_PLATFORM"
echo "Using Python at $(which python)"
echo "Using EMBODIED_PATH=$EMBODIED_PATH"
echo "Using EMBODIED_CONFIG_DIR=$EMBODIED_CONFIG_DIR"
echo "Using REPO_PATH=$REPO_PATH"
echo "Using CONFIG_DIR=$CONFIG_DIR"

LOG_BASE_DIR="${LOG_BASE_DIR:-${EMBODIED_PATH}/logs}"
LOG_DIR="${LOG_BASE_DIR}/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"

CMD=(
    python
    "$SRC_FILE"
    --config-path "$CONFIG_DIR"
    --config-name "$CONFIG_NAME"
    "runner.logger.log_path=${LOG_DIR}"
)

printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
