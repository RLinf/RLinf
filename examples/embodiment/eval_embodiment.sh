#! /bin/bash

# =============================================================================
# eval_embodiment.sh – RoboCasa365 evaluation launcher for headless servers
# =============================================================================
# This script wraps the RoboCasa365 evaluation pipeline with proper OSMesa
# (off-screen rendering) support. It is the evaluation counterpart of
# run_embodiment.sh and falls back gracefully when OSMesa is unavailable.
#
# Usage:
#   bash examples/embodiment/eval_embodiment.sh <config_name> [hydra_overrides...]
#
# Examples:
#   bash examples/embodiment/eval_embodiment.sh robocasa365_eval_openpi
#   bash examples/embodiment/eval_embodiment.sh robocasa365_eval_openpi \
#       env.eval.task_soup=composite_unseen env.eval.task_mode=composite
# =============================================================================

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_embodied_agent.py"

# ---------------------------------------------------------------------------
# Rendering backend selection (headless-first)
# ---------------------------------------------------------------------------
# OSMesa is preferred for headless servers because it renders entirely in
# CPU memory without requiring a display or GPU context.  EGL can also work
# headlessly on NVIDIA GPUs but may need a valid DISPLAY on some systems.
#
# Detection logic:
#   1. Honour explicit user overrides (MUJOCO_GL / PYOPENGL_PLATFORM).
#   2. If not set, prefer OSMesa when libOSMesa is available.
#   3. Fall back to EGL when OSMesa is missing.

detect_osmesa() {
    if ldconfig -p 2>/dev/null | grep -q libOSMesa || \
       find /usr/lib -name 'libOSMesa*' 2>/dev/null | grep -q .; then
        return 0
    fi
    return 1
}

if [[ -z "${MUJOCO_GL:-}" ]]; then
    if detect_osmesa; then
        export MUJOCO_GL="osmesa"
        echo "[eval_embodiment] auto-detected OSMesa → MUJOCO_GL=osmesa"
    else
        export MUJOCO_GL="egl"
        echo "[eval_embodiment] OSMesa not found, falling back to MUJOCO_GL=egl"
    fi
fi

if [[ -z "${PYOPENGL_PLATFORM:-}" ]]; then
    export PYOPENGL_PLATFORM="${MUJOCO_GL}"
    echo "[eval_embodiment] PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
fi

# ---------------------------------------------------------------------------
# Environment variables (mirror run_embodiment.sh for compatibility)
# ---------------------------------------------------------------------------
export ROBOTWIN_PATH=${ROBOTWIN_PATH:-"/path/to/RoboTwin"}
export PYTHONPATH=${REPO_PATH}:${ROBOTWIN_PATH}:${PYTHONPATH:-}

export OMNIGIBSON_NO_OMNI_LOGS=${OMNIGIBSON_NO_OMNI_LOGS:-1}
export OMNIGIBSON_DEBUG=${OMNIGIBSON_DEBUG:-0}
export OMNIGIBSON_DATA_PATH=${OMNIGIBSON_DATA_PATH:-}
export OMNIGIBSON_DATASET_PATH=${OMNIGIBSON_DATASET_PATH:-${OMNIGIBSON_DATA_PATH}/behavior-1k-assets/}
export OMNIGIBSON_KEY_PATH=${OMNIGIBSON_KEY_PATH:-${OMNIGIBSON_DATA_PATH}/omnigibson.key}
export OMNIGIBSON_ASSET_PATH=${OMNIGIBSON_ASSET_PATH:-${OMNIGIBSON_DATA_PATH}/omnigibson-robot-assets/}
export OMNIGIBSON_HEADLESS=${OMNIGIBSON_HEADLESS:-1}
export ISAAC_PATH=${ISAAC_PATH:-/path/to/isaac-sim}
export EXP_PATH=${EXP_PATH:-$ISAAC_PATH/apps}
export CARB_APP_PATH=${CARB_APP_PATH:-$ISAAC_PATH/kit}

export POLARIS_DATA_PATH=${POLARIS_DATA_PATH:-"/path/to/dataset/PolaRiS-Hub"}

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
if [ -z "${1:-}" ]; then
    echo "Usage: $0 <config_name> [hydra_overrides...]" >&2
    echo "Example: $0 robocasa365_eval_openpi" >&2
    exit 1
fi

CONFIG_NAME="$1"
shift

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "Using MUJOCO_GL=${MUJOCO_GL} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM}"
echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-eval-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/eval_embodiment.log"
mkdir -p "${LOG_DIR}"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo "${CMD} $@" > "${MEGA_LOG_FILE}"
${CMD} "$@" 2>&1 | tee -a "${MEGA_LOG_FILE}"
