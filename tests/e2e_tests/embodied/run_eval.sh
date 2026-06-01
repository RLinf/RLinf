#! /bin/bash

set -eo pipefail

CONFIG=$1
BACKEND=${2:-"egl"}
shift 2 || true

export MUJOCO_GL=${BACKEND}
export PYOPENGL_PLATFORM=${BACKEND}
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

python ${REPO_PATH}/examples/embodiment/eval_embodied_agent.py --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name ${CONFIG} "$@"
