#! /bin/bash
set -x

tabs 4
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

output=$(python ${REPO_PATH}/tools/auto_placement/scheduler_task.py --config-path $REPO_PATH/tests/e2e_tests/auto_placement  --config-name qwen2.5-1.5b-grpo)
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Python script execution failed with exit code: $exit_code"
    exit $exit_code
fi

EXPECTED_OUTPUT0='==================================================
Best placement for this task is:

cluster:
  num_nodes: 1
  num_gpus_per_node: 8
  component_placement:
    rollout,actor: all'

EXPECTED_OUTPUT1='==================================================
Best placement for this task is:

cluster:
  num_nodes: 1
  num_gpus_per_node: 8
  component_placement:
    actor,rollout: all'
if [ "$EXPECTED_OUTPUT0" = "$output" ] || [ "$EXPECTED_OUTPUT1" = "$output" ]; then
    echo "Output matches the expected result."
    exit 0
else
    exit 1
fi
