#! /bin/bash
set -x

tabs 4
export MUJOCO_GL="osmesa"
export PYOPENGL_PLATFORM="osmesa"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

python ${REPO_PATH}/examples/embodiment/train_embodied_agent.py --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name libero_130_grpo_openvlaoft