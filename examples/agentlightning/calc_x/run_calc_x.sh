#! /bin/bash
set -x

tabs 4
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname $(dirname "$CONFIG_PATH")))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${REPO_PATH}/examples:$PYTHONPATH

# Check if first argument is eval parameter (starts with "eval=")
if [[ "$1" == eval=* ]] || [[ "$1" == +eval=* ]]; then
    CONFIG_NAME="qwen2.5-1.5b-trajectory"
    # Process arguments: add + prefix if not present for eval and eval_checkpoint_dir
    # Also override resume_dir to null to avoid auto-loading checkpoint
    ARGS=()
    eval_checkpoint_dir=""
    for arg in "$@"; do
        if [[ "$arg" == eval=* ]] && [[ "$arg" != +eval=* ]]; then
            ARGS+=("+eval=true")
        elif [[ "$arg" == eval_checkpoint_dir=* ]] && [[ "$arg" != +eval_checkpoint_dir=* ]]; then
            eval_checkpoint_dir="${arg#eval_checkpoint_dir=}"
            ARGS+=("+$arg")
        elif [[ "$arg" == +eval_checkpoint_dir=* ]]; then
            eval_checkpoint_dir="${arg#+eval_checkpoint_dir=}"
            ARGS+=("$arg")
        else
            ARGS+=("$arg")
        fi
    done
    # Override resume_dir to null in eval mode to avoid auto-loading checkpoint
    ARGS+=("+runner.resume_dir=null")
    # If eval_checkpoint_dir is provided, set actor.model.megatron_checkpoint to the actor checkpoint path
    if [ -n "$eval_checkpoint_dir" ]; then
        # Find the actual checkpoint path (handle both directory and direct checkpoint path)
        if [[ "$(basename "$eval_checkpoint_dir")" == global_step_* ]]; then
            ckpt_path="$eval_checkpoint_dir"
        else
            # Find first global_step_* directory
            ckpt_path=$(find "$eval_checkpoint_dir" -maxdepth 1 -type d -name "global_step_*" | sort -V | head -n 1)
        fi
        if [ -n "$ckpt_path" ] && [ -d "$ckpt_path" ]; then
            actor_checkpoint_path="${ckpt_path}/actor"
            if [ -d "$actor_checkpoint_path" ]; then
                ARGS+=("+actor.model.megatron_checkpoint=${actor_checkpoint_path}")
            fi
        fi
    fi
    python ${CONFIG_PATH}/main.py --config-path ${CONFIG_PATH}/config/ --config-name $CONFIG_NAME "${ARGS[@]}"
else
    if [ -z "$1" ]; then
        CONFIG_NAME="qwen2.5-1.5b-trajectory"
    else
        CONFIG_NAME=$1
        shift
    fi
    python ${CONFIG_PATH}/main.py --config-path ${CONFIG_PATH}/config/ --config-name $CONFIG_NAME "$@"
fi

