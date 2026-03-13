#!/bin/bash
# Review and filter success/failure classifier data collected by
# collect_classifier_data.sh.
#
# Usage:
#   bash examples/embodiment/review_classifier_data.sh <log_dir>
#
# Examples:
#   bash examples/embodiment/review_classifier_data.sh \
#       logs/20260305-10:00:00-reward-classifier-dex_pnp

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/review_classifier_data.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

# Avoid X11 shared-memory errors in Docker containers with small /dev/shm.
export QT_X11_NO_MITSHM=1

if [ -z "$1" ]; then
    echo "Usage: bash examples/embodiment/review_classifier_data.sh <log_dir>"
    echo ""
    echo "Available log dirs:"
    ls -d "${REPO_PATH}"/logs/*reward-classifier* 2>/dev/null || echo "  (none found)"
    exit 1
fi

LOG_DIR="$1"

echo "Using Python at $(which python)"
echo "Reviewing: ${LOG_DIR}"
python "${SRC_FILE}" --log_dir "${LOG_DIR}"
