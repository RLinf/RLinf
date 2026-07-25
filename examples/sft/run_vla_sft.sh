#! /bin/bash
set -euo pipefail

export EMBODIED_PATH="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$(cd -- "${EMBODIED_PATH}/../.." && pwd)"
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"

export FASTWAM_PATH="${FASTWAM_PATH:-${REPO_PATH}/.venv/FastWAM}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_PATH}/checkpoints}"
export PYTHONPATH="${REPO_PATH}:${LIBERO_REPO_PATH:-}:${PYTHONPATH:-}"

export DREAMZERO_PATH="${DREAMZERO_PATH:-${REPO_PATH}/.venv/dreamzero}"
export PYTHONPATH="${DREAMZERO_PATH}:${PYTHONPATH}"

if [ "$#" -eq 0 ]; then
    CONFIG_NAME="maniskill_ppo_openvlaoft"
else
    CONFIG_NAME=$1
    shift
fi

cd "${REPO_PATH}"
echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
HYDRA_DATASET_OVERRIDES=()
if [[ -n "${FASTWAM_DATASET_DIRS:-}" ]]; then
    HYDRA_DATASET_OVERRIDES+=("data.train_data_paths=[${FASTWAM_DATASET_DIRS}]")
elif [[ -n "${FASTWAM_DATASET_DIR:-}" ]]; then
    # Preserve the existing one-suite override for smoke/debug runs.
    HYDRA_DATASET_OVERRIDES+=("data.train_data_paths=${FASTWAM_DATASET_DIR}")
fi
CMD=(python "${SRC_FILE}" --config-path "${EMBODIED_PATH}/config/" --config-name "${CONFIG_NAME}" "runner.logger.log_path=${LOG_DIR}" "${HYDRA_DATASET_OVERRIDES[@]}" "$@")
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"