#!/usr/bin/env bash
# Calculate OpenPI normalization statistics for a local LeRobot dataset.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "${SCRIPT_DIR}/.." && pwd)

DEFAULT_DATASET="${REPO}/data/lerobot-data_mixed_8_v30"
DATASET_PATH="${1:-${OPENPI_DATASET_PATH:-${DEFAULT_DATASET}}}"
CONFIG_NAME="${2:-${OPENPI_CONFIG_NAME:-pi05_aloha_robotwin}}"
VENV_PATH="${OPENPI_VENV_PATH:-${REPO}/.venv}"
PROGRESS_INTERVAL_SECONDS="${PROGRESS_INTERVAL_SECONDS:-30}"

usage() {
    cat <<EOF
Usage: $(basename "$0") [DATASET_PATH] [CONFIG_NAME]

Calculate OpenPI normalization statistics for a local LeRobot v3 dataset.

Arguments:
  DATASET_PATH  Dataset root containing meta/info.json.
                Default: ${DEFAULT_DATASET}
  CONFIG_NAME   OpenPI data configuration name.
                Default: pi05_aloha_robotwin

Environment overrides:
  OPENPI_DATASET_PATH          Alternative to positional DATASET_PATH.
  OPENPI_CONFIG_NAME           Alternative to positional CONFIG_NAME.
  OPENPI_VENV_PATH             Python virtualenv. Default: ${REPO}/.venv
  PROGRESS_INTERVAL_SECONDS    Hidden index-stage status interval. Default: 30

Examples:
  $(basename "$0")
  $(basename "$0") /path/to/lerobot-v3 pi05_aloha_robotwin
  OPENPI_DATASET_PATH=/path/to/lerobot-v3 $(basename "$0")
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ ! "${PROGRESS_INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[norm-stats] ERROR: PROGRESS_INTERVAL_SECONDS must be a positive integer." >&2
    exit 2
fi

if [[ "${DATASET_PATH}" != /* ]]; then
    DATASET_PATH="${REPO}/${DATASET_PATH}"
fi
if [[ ! -f "${DATASET_PATH}/meta/info.json" ]]; then
    echo "[norm-stats] ERROR: Missing ${DATASET_PATH}/meta/info.json" >&2
    exit 1
fi
if [[ ! -x "${VENV_PATH}/bin/python" ]]; then
    echo "[norm-stats] ERROR: Python not found at ${VENV_PATH}/bin/python" >&2
    exit 1
fi

cd "${REPO}"
source "${VENV_PATH}/bin/activate"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

TOTAL_FILES=$(find "${DATASET_PATH}/data" -type f -name 'file-*.parquet' | wc -l)
if [[ "${TOTAL_FILES}" -eq 0 ]]; then
    echo "[norm-stats] ERROR: No v3 Parquet files found under ${DATASET_PATH}/data" >&2
    exit 1
fi

# Prevent duplicate runs launched through this reusable script.
exec 9>"${DATASET_PATH}/.calculate_norm_stats.lock"
if command -v flock >/dev/null 2>&1 && ! flock -n 9; then
    echo "[norm-stats] ERROR: Another norm-stats run holds the dataset lock." >&2
    exit 1
fi

echo "[norm-stats] Repository : ${REPO}"
echo "[norm-stats] Dataset    : ${DATASET_PATH}"
echo "[norm-stats] Config     : ${CONFIG_NAME}"
echo "[norm-stats] Files      : ${TOTAL_FILES}"
echo "[norm-stats] Output     : ${DATASET_PATH}/norm_stats.json"

"${VENV_PATH}/bin/python" toolkits/lerobot/calculate_norm_stats.py \
    --config-name "${CONFIG_NAME}" \
    --repo-id "${DATASET_PATH}" &
WORKER_PID=$!

monitor_index_progress() {
    local current_path chunk_name file_name chunk_index file_index completed
    while kill -0 "${WORKER_PID}" 2>/dev/null; do
        current_path=$(
            readlink /proc/"${WORKER_PID}"/fd/* 2>/dev/null |
                grep "${DATASET_PATH}/data/chunk-[0-9]*/file-[0-9]*\.parquet" |
                tail -1 || true
        )
        if [[ -n "${current_path}" ]]; then
            chunk_name=$(basename "$(dirname "${current_path}")")
            file_name=$(basename "${current_path}" .parquet)
            chunk_index=${chunk_name#chunk-}
            file_index=${file_name#file-}
            completed=$((10#${chunk_index} * 1000 + 10#${file_index} + 1))
            if ((completed > TOTAL_FILES)); then
                completed=${TOTAL_FILES}
            fi
            awk -v done="${completed}" -v total="${TOTAL_FILES}" \
                'BEGIN {printf "[norm-stats] Building dataset index: %d/%d (%.1f%%)\n", done, total, done*100/total}'
        else
            echo "[norm-stats] Waiting for index I/O or computing statistics; see the Python tqdm output."
        fi
        sleep "${PROGRESS_INTERVAL_SECONDS}"
    done
}

monitor_index_progress &
MONITOR_PID=$!

cleanup_monitor() {
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup_monitor EXIT

set +e
wait "${WORKER_PID}"
STATUS=$?
set -e

cleanup_monitor
trap - EXIT

if [[ "${STATUS}" -ne 0 ]]; then
    echo "[norm-stats] ERROR: Calculation failed with exit code ${STATUS}." >&2
    exit "${STATUS}"
fi

echo "[norm-stats] Complete: ${DATASET_PATH}/norm_stats.json"
