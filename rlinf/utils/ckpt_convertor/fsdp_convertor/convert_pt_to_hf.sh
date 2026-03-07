#!/bin/bash
# Convert FSDP checkpoint to HuggingFace SafeTensors format (action model)
#
# Usage:
#   bash convert_pt_to_hf.sh [CONFIG_NAME] [HYDRA_OVERRIDES...]
#
# Examples:
#   bash convert_pt_to_hf.sh
#   bash convert_pt_to_hf.sh fsdp_model_convertor
#   bash convert_pt_to_hf.sh fsdp_model_convertor convertor.ckpt_path=/path/to/ckpt

set -e

export FSDP_CONVERTOR_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname $(dirname "$FSDP_CONVERTOR_PATH"))))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Activate the openpi environment
source switch_env openpi 2>/dev/null || true

CONFIG_NAME=${1:-fsdp_model_convertor}
shift 2>/dev/null || true

python ${FSDP_CONVERTOR_PATH}/convert_pt_to_hf.py \
    --config-path ${FSDP_CONVERTOR_PATH}/config \
    --config-name ${CONFIG_NAME} \
    "$@"
