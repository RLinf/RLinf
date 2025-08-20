#! /bin/bash
set -x

tabs 4

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
unset HOME # GitHub action sets HOME to a wrong path (/github/home), breaking simulator

python ${REPO_PATH}/examples/embodiment/train_embodied_agent.py --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name ppo_openvla