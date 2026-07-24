#!/usr/bin/env bash
set -euo pipefail

# Reproduce the pi0.5 SFT flow for ocl-data/lerobot-data/sandwich_new_all.
#
# Common usage:
#   bash examples/sft/run_sandwich_new_all_pi05_sft.sh
#   SMOKE=1 bash examples/sft/run_sandwich_new_all_pi05_sft.sh
#   RUN_STEPS=5000 SAVE_INTERVAL=1000 bash examples/sft/run_sandwich_new_all_pi05_sft.sh
#   RESTART=1 bash examples/sft/run_sandwich_new_all_pi05_sft.sh
#   DRY_RUN=1 bash examples/sft/run_sandwich_new_all_pi05_sft.sh

EMBODIED_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd "${EMBODIED_PATH}/../.." && pwd)"
SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

CONFIG_NAME="${CONFIG_NAME:-sandwich_new_all_sft_openpi_pi05}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_PATH}/.venv/bin/python}"
RAY_BIN="${RAY_BIN:-${REPO_PATH}/.venv/bin/ray}"

DATASET_PATH="${DATASET_PATH:-${REPO_PATH}/ocl-data/lerobot-data/sandwich_new_all}"
OPENPI_ROOT="${OPENPI_ROOT:-/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi}"
JAX_PI05_CKPT="${JAX_PI05_CKPT:-${OPENPI_ROOT}/models/pi05_base}"
TORCH_PI05_CKPT="${TORCH_PI05_CKPT:-${OPENPI_ROOT}/checkpoints/torch/pi05_sandwich_new_all_rlinf_base}"
ASSET_ID="${ASSET_ID:-pi05_sandwich_new_all}"
NORM_STATS_SRC="${NORM_STATS_SRC:-${OPENPI_ROOT}/assets/${ASSET_ID}/${ASSET_ID}/norm_stats.json}"

# The ffmpeg used successfully for this run is in this conda env. Only PATH is
# prepended to avoid leaking conda shared libraries into the RLinf venv.
CONDA_FFMPEG_ENV="${CONDA_FFMPEG_ENV:-/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun-home/miniforge3/envs/yushun-openpi}"

RAY_SESSION="${RAY_SESSION:-rlinf_ray_sft}"
TRAIN_SESSION="${TRAIN_SESSION:-rlinf_sandwich_pi05_sft}"
RAY_ADDRESS="${RAY_ADDRESS:-127.0.0.1:6379}"
RAY_NODE_IP="${RAY_NODE_IP:-127.0.0.1}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-1}"

RUN_STEPS="${RUN_STEPS:-20000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-sandwich_new_all_pi05_sft}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
RESTART="${RESTART:-0}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE="${SMOKE:-0}"
SMOKE_STEPS="${SMOKE_STEPS:-1}"

if [[ "${SMOKE}" == "1" ]]; then
  RUN_STEPS="${SMOKE_STEPS}"
  SAVE_INTERVAL="1"
  EXPERIMENT_NAME="${EXPERIMENT_NAME}_smoke"
  TRAIN_SESSION="${TRAIN_SESSION}_smoke"
fi

RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_PATH}/logs/${CONFIG_NAME}_$(date +%Y%m%d-%H%M%S)}"
RUN_LOG_FILE="${RUN_LOG_FILE:-${RUN_LOG_DIR}/run.log}"

die() {
  echo "error: $*" >&2
  exit 1
}

quote_args() {
  printf "%q " "$@"
}

ray_status() {
  if command -v timeout >/dev/null 2>&1; then
    timeout 10 env RAY_ADDRESS="${RAY_ADDRESS}" "${RAY_BIN}" status >/dev/null 2>&1
  else
    env RAY_ADDRESS="${RAY_ADDRESS}" "${RAY_BIN}" status >/dev/null 2>&1
  fi
}

configure_env() {
  if [[ -d "${CONDA_FFMPEG_ENV}/bin" ]]; then
    export PATH="${CONDA_FFMPEG_ENV}/bin:${PATH}"
  fi

  export EMBODIED_PATH
  export REPO_PATH
  export SRC_FILE
  export RAY_ADDRESS
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

  if [[ -n "${PYTHONPATH:-}" ]]; then
    export PYTHONPATH="${REPO_PATH}:${PYTHONPATH}"
  else
    export PYTHONPATH="${REPO_PATH}"
  fi
}

ensure_prereqs() {
  if [[ "${INSTALL_DEPS}" == "1" ]]; then
    bash "${REPO_PATH}/requirements/install.sh" embodied \
      --model openpi \
      --env maniskill_libero \
      --no-root \
      --no-flash-attn
  fi

  [[ -x "${PYTHON_BIN}" ]] || die "missing ${PYTHON_BIN}; run INSTALL_DEPS=1 $0 first, or run requirements/install.sh manually"
  [[ -x "${RAY_BIN}" ]] || die "missing ${RAY_BIN}; check the RLinf .venv install"
  [[ -f "${SRC_FILE}" ]] || die "missing ${SRC_FILE}"
  [[ -d "${DATASET_PATH}" ]] || die "missing dataset: ${DATASET_PATH}"
  command -v tmux >/dev/null 2>&1 || die "tmux is required for Ray/training sessions"
}

ensure_huggingface_hub_pin() {
  local current_version
  current_version="$("${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    print(version("huggingface-hub"))
except PackageNotFoundError:
    print("")
PY
)"

  if [[ "${current_version}" == "0.36.2" ]]; then
    return
  fi

  echo "Pinning huggingface-hub to 0.36.2 for transformers compatibility (current: ${current_version:-missing})"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "${PYTHON_BIN}" "huggingface-hub==0.36.2"
  else
    "${PYTHON_BIN}" -m pip install "huggingface-hub==0.36.2"
  fi
}

ensure_checkpoint() {
  if [[ ! -f "${TORCH_PI05_CKPT}/model.safetensors" ]]; then
    [[ -d "${JAX_PI05_CKPT}" ]] || die "missing JAX pi0.5 checkpoint: ${JAX_PI05_CKPT}"
    mkdir -p "$(dirname "${TORCH_PI05_CKPT}")"
    echo "Converting JAX pi0.5 checkpoint to RLinf PyTorch checkpoint: ${TORCH_PI05_CKPT}"
    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
      "${PYTHON_BIN}" "${REPO_PATH}/rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py" \
        --checkpoint-dir "${JAX_PI05_CKPT}" \
        --config-name pi05_aloha \
        --output-path "${TORCH_PI05_CKPT}" \
        --precision bfloat16
  fi

  local norm_stats_dst="${TORCH_PI05_CKPT}/${ASSET_ID}/norm_stats.json"
  if [[ ! -f "${norm_stats_dst}" ]]; then
    [[ -f "${NORM_STATS_SRC}" ]] || die "missing norm stats: ${NORM_STATS_SRC}"
    mkdir -p "$(dirname "${norm_stats_dst}")"
    cp "${NORM_STATS_SRC}" "${norm_stats_dst}"
  fi
}

start_ray() {
  if ray_status; then
    echo "Ray is already available at ${RAY_ADDRESS}"
    return
  fi

  if tmux has-session -t "${RAY_SESSION}" 2>/dev/null; then
    die "tmux session ${RAY_SESSION} exists but Ray is not ready; inspect it with: tmux attach -t ${RAY_SESSION}"
  fi

  mkdir -p "${REPO_PATH}/logs"
  local ray_log="${REPO_PATH}/logs/ray_sft_$(date +%Y%m%d-%H%M%S).log"
  local ray_cmd
  ray_cmd="cd $(printf "%q" "${REPO_PATH}") && $(printf "%q" "${RAY_BIN}") start --head --node-ip-address=$(printf "%q" "${RAY_NODE_IP}") --num-gpus=$(printf "%q" "${RAY_NUM_GPUS}") --include-dashboard=false --block 2>&1 | tee $(printf "%q" "${ray_log}")"

  echo "Starting Ray in tmux session ${RAY_SESSION}"
  tmux new-session -d -s "${RAY_SESSION}" "${ray_cmd}"

  for _ in $(seq 1 60); do
    if ray_status; then
      echo "Ray is ready at ${RAY_ADDRESS}"
      return
    fi
    sleep 1
  done

  die "Ray did not become ready; inspect logs with: tmux attach -t ${RAY_SESSION}"
}

build_train_cmd() {
  TRAIN_CMD=(
    "${PYTHON_BIN}"
    "${SRC_FILE}"
    --config-name
    "${CONFIG_NAME}"
    "runner.logger.log_path=${RUN_LOG_DIR}"
    "runner.logger.experiment_name=${EXPERIMENT_NAME}"
    "runner.max_steps=${RUN_STEPS}"
    "runner.save_interval=${SAVE_INTERVAL}"
    "actor.optim.total_training_steps=${RUN_STEPS}"
    "data.train_data_paths=${DATASET_PATH}"
    "actor.model.model_path=${TORCH_PI05_CKPT}"
    "actor.openpi_data.repo_id=${ASSET_ID}"
    "actor.model.openpi_data.repo_id=${ASSET_ID}"
  )
}

start_training() {
  mkdir -p "${RUN_LOG_DIR}"

  local quoted_cmd
  quoted_cmd="$(quote_args "${TRAIN_CMD[@]}")"

  {
    echo "# started: $(date -Is)"
    echo "# repo: ${REPO_PATH}"
    echo "# command:"
    echo "${quoted_cmd}"
  } >"${RUN_LOG_FILE}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN=1; no process was started."
    echo "Training command:"
    echo "${quoted_cmd}"
    return
  fi

  if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    if [[ "${RESTART}" == "1" ]]; then
      tmux kill-session -t "${TRAIN_SESSION}"
    else
      echo "Training tmux session already exists: ${TRAIN_SESSION}"
      echo "Attach with: tmux attach -t ${TRAIN_SESSION}"
      echo "Use RESTART=1 to kill that session and start a new run."
      return
    fi
  fi

  local tmux_cmd
  tmux_cmd="cd $(printf "%q" "${REPO_PATH}") && export PATH=$(printf "%q" "${PATH}") PYTHONPATH=$(printf "%q" "${PYTHONPATH}") EMBODIED_PATH=$(printf "%q" "${EMBODIED_PATH}") REPO_PATH=$(printf "%q" "${REPO_PATH}") SRC_FILE=$(printf "%q" "${SRC_FILE}") RAY_ADDRESS=$(printf "%q" "${RAY_ADDRESS}") MUJOCO_GL=$(printf "%q" "${MUJOCO_GL}") PYOPENGL_PLATFORM=$(printf "%q" "${PYOPENGL_PLATFORM}") HYDRA_FULL_ERROR=$(printf "%q" "${HYDRA_FULL_ERROR}") && ${quoted_cmd} 2>&1 | tee -a $(printf "%q" "${RUN_LOG_FILE}")"

  tmux new-session -d -s "${TRAIN_SESSION}" "${tmux_cmd}"

  echo "Started training in tmux session: ${TRAIN_SESSION}"
  echo "Attach: tmux attach -t ${TRAIN_SESSION}"
  echo "Log: ${RUN_LOG_FILE}"
  echo "TensorBoard log root: ${RUN_LOG_DIR}"
}

main() {
  configure_env
  build_train_cmd

  if [[ "${DRY_RUN}" == "1" ]]; then
    start_training
    return
  fi

  ensure_prereqs
  ensure_huggingface_hub_pin
  ensure_checkpoint
  start_ray
  start_training
}

main "$@"
