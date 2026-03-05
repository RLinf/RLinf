#!/bin/bash
# Convert FSDP checkpoint to HuggingFace SafeTensors format (value model)
#
# Usage:
#   bash convert_pt_to_hf_vla_lib.sh [HYDRA_OVERRIDES...]
#
# Examples:
#   bash convert_pt_to_hf_vla_lib.sh
#   bash convert_pt_to_hf_vla_lib.sh convertor.ckpt_path=/path/to/ckpt convertor.save_path=/path/to/output

set -e

export FSDP_CONVERTOR_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname $(dirname "$FSDP_CONVERTOR_PATH"))))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Activate the openpi environment
source switch_env openpi 2>/dev/null || true

python ${FSDP_CONVERTOR_PATH}/convert_pt_to_hf.py \
    --config-path ${FSDP_CONVERTOR_PATH}/config \
    --config-name fsdp_vla_lib_model_convertor \
    "$@"
