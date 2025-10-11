#! /bin/bash
set -x

tabs 4

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

python ${REPO_PATH}/examples/embodiment/train_embodied_agent.py --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name ppo_openvla