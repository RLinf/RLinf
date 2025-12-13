#! /bin/bash
set -x

tabs 4

CONFIG=$1
BACKEND=${2:-"egl"}

export MUJOCO_GL=${BACKEND}
export PYOPENGL_PLATFORM=${BACKEND}
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

python ${REPO_PATH}/examples/sft/train_embodied_sft.py --config-path ${REPO_PATH}/tests/e2e_tests/sft --config-name ${CONFIG}