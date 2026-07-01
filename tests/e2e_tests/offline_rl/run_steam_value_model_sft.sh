#! /bin/bash
set -x

tabs 4

CONFIG=$1
BACKEND=${2:-"egl"}

export MUJOCO_GL=${BACKEND}
export PYOPENGL_PLATFORM=${BACKEND}
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

E2E_CONFIG="${REPO_PATH}/tests/e2e_tests/offline_rl"

python "${REPO_PATH}/examples/offline_rl/advantage_labeling/steam/train_steam.py" \
  --config-path "${E2E_CONFIG}" \
  --config-name "${CONFIG}"
