#! /bin/bash
set -x

tabs 4

export FSDP_CONVERTOR_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname $(dirname "$FSDP_CONVERTOR_PATH")))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH


python ${REPO_PATH}/toolkits/ckpt_convertor/fsdp_convertor/convert_pt_to_hf.py --config-path ${REPO_PATH}/toolkits/ckpt_convertor/fsdp_convertor/config --config-name fsdp_model_convertor