#! /bin/bash
set -x

tabs 4

# Default REPO_PATH to repo root when not set (e.g. local run)
REPO_PATH=${REPO_PATH:-$(cd "$(dirname "$0")/../../.." && pwd)}
export REPO_PATH

CONFIG=$1
BACKEND=${2:-"egl"}
shift 2 2>/dev/null || shift 1 2>/dev/null || true

export MUJOCO_GL=${BACKEND}
export PYOPENGL_PLATFORM=${BACKEND}
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

python ${REPO_PATH}/examples/embodiment/train_offlinerl.py --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name ${CONFIG} "$@"
