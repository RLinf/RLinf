#! /bin/bash
set -x

tabs 4
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export LIBERO_REPO_PATH="/workspace/libero"
export LIBERO_CONFIG_PATH=${LIBERO_REPO_PATH}
export PYTHONPATH=${LIBERO_REPO_PATH}:$PYTHONPATH
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
unset HOME # GitHub action sets HOME to a wrong path (/github/home), breaking simulator

python ${REPO_PATH}/examples/embodiment/train_embodied_agent.py --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name libero_130_grpo_openvlaoft