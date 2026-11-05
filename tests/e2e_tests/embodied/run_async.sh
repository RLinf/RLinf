#! /bin/bash
set -x

tabs 4

CONFIG=$1
BACKEND=${2:-"egl"}

export MUJOCO_GL=${BACKEND}
export PYOPENGL_PLATFORM=${BACKEND}
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
# setup_sim_env is defined in sim_env.sh.
source "${REPO_PATH}/examples/embodiment/sim_env.sh"
setup_sim_env

python ${REPO_PATH}/examples/embodiment/train_async.py --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name ${CONFIG}