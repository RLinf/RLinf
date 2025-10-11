#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    echo "Please provide a config name as the first argument."
    exit 1
else
    CONFIG_NAME=$1
fi
python ${REPO_PATH}/examples/reasoning/main_grpo.py --config-path $REPO_PATH/tests/e2e_tests/reasoning/  --config-name $CONFIG_NAME