#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/train_supermimic.py"
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

CONFIG_NAME=${1:-supermimic_smoke}

# Only the local smoke launch loosens Ray's host-memory monitor. A normal
# training config keeps Ray's own threshold. Set the variable yourself to
# override this smoke default.
if [ "${CONFIG_NAME}" = "supermimic_smoke" ] && [ -z "${RAY_memory_usage_threshold:-}" ]; then
    export RAY_memory_usage_threshold=0.99
    echo "Smoke launch sets RAY_memory_usage_threshold=0.99. Other configs do not."
fi

python "${SRC_FILE}" --config-name "${CONFIG_NAME}"
