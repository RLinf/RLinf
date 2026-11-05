#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_supermimic.py"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# No arguments keeps the original small smoke workflow; an explicit config
# runs its configured sizes. Small runs use the same policy as the main recipe.
SMOKE_DEFAULT=0
if [ "$#" -eq 0 ]; then
    SMOKE_DEFAULT=1
    set -- supermimic_lafan1_zita env.num_envs=8 env.num_clips=1 env.replay_length=64 \
        algorithm.total_updates=3 algorithm.wm_only_updates=0 \
        algorithm.wm_window_batch=64 algorithm.imagine_batch=8
fi
CONFIG_NAME=$1
shift $(( $# > 0 ? 1 : 0 ))
EXTRA_ARGS=("$@")

# Only the local smoke launch loosens Ray's host-memory monitor. A normal
# training config keeps Ray's own threshold. Set the variable yourself to
# override this smoke default.
if [ "$SMOKE_DEFAULT" -eq 1 ] && [ -z "${RAY_memory_usage_threshold:-}" ]; then
    export RAY_memory_usage_threshold=0.99
    echo "Smoke launch sets RAY_memory_usage_threshold=0.99. Other configs do not."
fi

# Like run_embodiment.sh: each launch gets its own log_path unless the caller
# passes one (for example to resume into an existing run with resume_dir=auto).
LOG_OVERRIDE=()
if ! printf '%s\n' "${EXTRA_ARGS[@]}" | grep -q '^runner\.logger\.log_path='; then
    LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${CONFIG_NAME}"
    mkdir -p "${LOG_DIR}"
    LOG_OVERRIDE=("runner.logger.log_path=${LOG_DIR}")
    echo "log_path=${LOG_DIR}"
fi

python "${SRC_FILE}" --config-name "${CONFIG_NAME}" "${LOG_OVERRIDE[@]}" "${EXTRA_ARGS[@]}"
