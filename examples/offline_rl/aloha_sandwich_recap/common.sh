#!/usr/bin/env bash

if [[ -n "${ALOHA_SANDWICH_RECAP_COMMON_SH:-}" ]]; then
  return 0
fi
ALOHA_SANDWICH_RECAP_COMMON_SH=1

ALOHA_RECAP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "${ALOHA_RECAP_SCRIPT_DIR}/../../.." && pwd)"
CONFIG_DIR="${REPO}/examples/offline_rl/config"
PY="${PY:-${REPO}/.venv/bin/python}"
OPENPI_ROOT="${OPENPI_ROOT:-/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi}"

RAW_DATA="${RAW_DATA:-/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_rl}"
DATASET="${DATASET:-/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/aloha-data/sandwich_lerobot_v21}"
POLICY_JAX_CKPT="${POLICY_JAX_CKPT:-/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/pi05_sandwich_new_all/pi05_sandwich_new_all_20260628_193430/49999}"
POLICY_CKPT="${POLICY_CKPT:-/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/checkpoints/torch/pi05_sandwich_new_all_20260628_193430_49999}"
POLICY_CONFIG_NAME="${POLICY_CONFIG_NAME:-pi05_sandwich_new_all}"
POLICY_PRECISION="${POLICY_PRECISION:-bfloat16}"
RUN_ROOT="${RUN_ROOT:-${REPO}/logs/aloha_sandwich_recap/$(date +'%Y%m%d-%H%M%S')}"

VALUE_STEPS="${VALUE_STEPS:-3000}"
CFG_STEPS="${CFG_STEPS:-3000}"
VALUE_MICRO_BATCH_SIZE="${VALUE_MICRO_BATCH_SIZE:-4}"
VALUE_GLOBAL_BATCH_SIZE="${VALUE_GLOBAL_BATCH_SIZE:-16}"
ADV_BATCH_SIZE="${ADV_BATCH_SIZE:-16}"
ADV_FLUSH_INTERVAL="${ADV_FLUSH_INTERVAL:-16}"

VALUE_RUN_DIR="${VALUE_RUN_DIR:-${RUN_ROOT}/value_sft}"
CFG_RUN_DIR="${CFG_RUN_DIR:-${RUN_ROOT}/cfg_rl}"
PIPELINE_LOG="${PIPELINE_LOG:-${RUN_ROOT}/pipeline.log}"
RETURNS="${RETURNS:-${DATASET}/meta/returns_sandwich_fail300.parquet}"
ADVANTAGES="${ADVANTAGES:-${DATASET}/meta/advantages_sandwich_fail300_N10_q30_teleop.parquet}"
VALUE_CKPT="${VALUE_CKPT:-${VALUE_RUN_DIR}/aloha_sandwich_value/checkpoints/global_step_${VALUE_STEPS}/actor}"

export REPO CONFIG_DIR PY OPENPI_ROOT RAW_DATA DATASET POLICY_JAX_CKPT POLICY_CKPT
export POLICY_CONFIG_NAME POLICY_PRECISION RUN_ROOT
export VALUE_STEPS CFG_STEPS VALUE_MICRO_BATCH_SIZE VALUE_GLOBAL_BATCH_SIZE
export ADV_BATCH_SIZE ADV_FLUSH_INTERVAL VALUE_RUN_DIR CFG_RUN_DIR
export PIPELINE_LOG RETURNS ADVANTAGES VALUE_CKPT

cd "${REPO}"
mkdir -p "${RUN_ROOT}" "${VALUE_RUN_DIR}" "${CFG_RUN_DIR}"

export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export REPO_PATH="${REPO}"
export PYTHONPATH="${REPO}:${OPENPI_ROOT}/src:${LIBERO_REPO_PATH:-}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export AV_LOG_FORCE_NOCOLOR="${AV_LOG_FORCE_NOCOLOR:-1}"
export LIBAV_LOG_LEVEL="${LIBAV_LOG_LEVEL:-quiet}"
export OPENCV_LOG_LEVEL="${OPENCV_LOG_LEVEL:-off}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HOME}/.cache/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "${PIPELINE_LOG}"
}

run_logged() {
  local name="$1"
  local logfile="$2"
  shift 2

  log "START ${name}"
  log "LOG ${logfile}"
  local errexit_set=0
  case $- in
    *e*)
      errexit_set=1
      set +e
      ;;
  esac
  {
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
    "$@"
  } 2>&1 | tee "${logfile}"

  local status=${PIPESTATUS[0]}
  if [[ "${errexit_set}" -eq 1 ]]; then
    set -e
  fi
  log "END ${name} status=${status}"
  return "${status}"
}

require_python() {
  if [[ ! -x "${PY}" ]]; then
    log "ERROR Python executable not found or not executable: ${PY}"
    exit 1
  fi
}

copy_policy_norm_stats() {
  local norm_stats_src="${POLICY_JAX_CKPT}/assets/${POLICY_CONFIG_NAME}/norm_stats.json"
  local norm_stats_dst="${POLICY_CKPT}/${POLICY_CONFIG_NAME}/norm_stats.json"

  if [[ -f "${norm_stats_dst}" ]]; then
    return 0
  fi
  if [[ ! -f "${norm_stats_src}" ]]; then
    log "ERROR policy norm stats missing: ${norm_stats_src}"
    exit 1
  fi

  mkdir -p "$(dirname "${norm_stats_dst}")"
  cp "${norm_stats_src}" "${norm_stats_dst}"
}

require_policy_checkpoint() {
  if [[ -f "${POLICY_CKPT}/model.safetensors" ]]; then
    copy_policy_norm_stats
    return 0
  fi

  if [[ ! -d "${POLICY_JAX_CKPT}/params" ]]; then
    log "ERROR policy checkpoint missing: ${POLICY_CKPT}/model.safetensors"
    log "ERROR JAX policy checkpoint params missing: ${POLICY_JAX_CKPT}/params"
    exit 1
  fi

  mkdir -p "$(dirname "${POLICY_CKPT}")"
  log "Converting policy JAX checkpoint: ${POLICY_JAX_CKPT}"
  log "Converted policy checkpoint: ${POLICY_CKPT}"
  run_logged "convert_policy_checkpoint" "${RUN_ROOT}/convert_policy_checkpoint.log" \
    bash -c 'export JAX_PLATFORMS=cpu; export CUDA_VISIBLE_DEVICES=; exec "$@"' bash \
    "${PY}" "${REPO}/rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py" \
    --checkpoint-dir "${POLICY_JAX_CKPT}" \
    --config-name "${POLICY_CONFIG_NAME}" \
    --output-path "${POLICY_CKPT}" \
    --precision "${POLICY_PRECISION}"

  if [[ ! -f "${POLICY_CKPT}/model.safetensors" ]]; then
    log "ERROR conversion did not create ${POLICY_CKPT}/model.safetensors"
    exit 1
  fi
  copy_policy_norm_stats
}

require_value_checkpoint() {
  if [[ ! -f "${VALUE_CKPT}/model_state_dict/full_weights.pt" ]]; then
    log "ERROR value checkpoint missing: ${VALUE_CKPT}/model_state_dict/full_weights.pt"
    exit 1
  fi
}

log_pipeline_header() {
  log "ALOHA sandwich RECAP pipeline"
  log "Repo: ${REPO}"
  log "Raw HITL data: ${RAW_DATA}"
  log "LeRobot dataset: ${DATASET}"
  log "Policy JAX checkpoint: ${POLICY_JAX_CKPT}"
  log "Policy checkpoint: ${POLICY_CKPT}"
  log "Run root: ${RUN_ROOT}"
  log "Python: ${PY}"
}
