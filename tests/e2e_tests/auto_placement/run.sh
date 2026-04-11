#! /bin/bash
set -x

tabs 4
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

output=$(python ${REPO_PATH}/toolkits/auto_placement/auto_placement_worker.py --config-path $REPO_PATH/tests/e2e_tests/auto_placement  --config-name qwen2.5-1.5b-grpo)
exit_code=$?

if [ $exit_code -ne 0 ]; then
    echo "Python script execution failed with exit code: $exit_code"
    exit $exit_code
fi


if echo "$output" | grep -q "actor, rollout : all" || echo "$output" | grep -q "rollout, actor : all"; then
    exit 0
else
    exit 1
fi