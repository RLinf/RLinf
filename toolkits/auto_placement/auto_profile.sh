#! /bin/bash
# auto profile (if needed) + auto placement for embodied training.
# Requires Ray to be running (e.g. ray start --head).
set -x

tabs 4

CONFIG_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../examples/embodiment" && pwd)"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export EMBODIED_PATH="${REPO_PATH}/examples/embodiment"

CONFIG_NAME="${1:-maniskill_ppo_openvlaoft_quickstart}"

python ${REPO_PATH}/toolkits/auto_placement/collect_profile.py \
    --config-path ${EMBODIED_PATH}/config \
    --config-name $CONFIG_NAME \