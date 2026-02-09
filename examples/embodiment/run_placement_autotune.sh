#! /bin/bash
set -x

tabs 4

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

## newly added
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
##

if [ -z "$1" ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft_quickstart"
else
    CONFIG_NAME=$1
fi


python ${REPO_PATH}/toolkits/auto_placement/auto_placement_worker.py \
    --config-path ${EMBODIED_PATH}/config \
    --config-name $CONFIG_NAME \