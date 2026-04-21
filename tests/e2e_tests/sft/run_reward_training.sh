#! /bin/bash
set -x

tabs 4

CONFIG=$1

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

python ${REPO_PATH}/examples/reward/train_reward_model.py --config-path ${REPO_PATH}/tests/e2e_tests/sft --config-name ${CONFIG}

