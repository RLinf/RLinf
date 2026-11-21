#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${RLINF_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
export RLINF_ROOT="${ROOT_DIR}"
export EMBODIED_PATH="${ROOT_DIR}/examples/embodiment"
export PYTHONPATH="${ROOT_DIR}${EXTRA_GPU_SITE_PACKAGES:+:${EXTRA_GPU_SITE_PACKAGES}}${PYTHONPATH:+:${PYTHONPATH}}"
export RLINF_SKIP_REALWORLD_SETUP=1
export RLINF_REALWORLD_PLATFORM=tianji

if [[ -z "${TIANJI_STAGE2_CHUNK:-}" ]]; then
  eval "$("${TIANJI_CPU_PYTHON:-${RLINF_PYTHON:-${PYTHON_BIN:-python3}}}" "${SCRIPT_DIR}/tianji_schedule_env.py")"
fi
unset RLT_ACTION_ENV_SCALE RLT_ACTION_ENV_BIAS
unset RLINF_TIANJI_STAGE1_ONLY

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "Set CUDA_VISIBLE_DEVICES=0,1; async Stage2 requires two GPUs." >&2
  exit 2
fi

IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#_visible_gpus[@]} < 2 )); then
  echo "Async Stage2 requires two visible GPUs: GPU0 for rollout/VLA+RLT and GPU1 for actor-critic training." >&2
  exit 2
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  _physical_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null \
    | sed '/^[[:space:]]*$/d' | wc -l | tr -d ' ')"
  if [[ -n "${_physical_gpu_count}" ]] && (( _physical_gpu_count < 2 )); then
    echo "Async Stage2 requires two physical GPUs; nvidia-smi detected ${_physical_gpu_count}." >&2
    exit 2
  fi
fi

if [[ -z "${TIANJI_RLT_STAGE1_MODEL_PATH:-}" ]]; then
  echo "Set TIANJI_RLT_STAGE1_MODEL_PATH (or TIANJI_MODEL_DIR) to the frozen Stage1 VLA/RLT checkpoint before async Stage2." >&2
  exit 2
fi
if [[ ! -d "${TIANJI_RLT_STAGE1_MODEL_PATH}" ]]; then
  echo "Stage1 checkpoint directory does not exist: ${TIANJI_RLT_STAGE1_MODEL_PATH}" >&2
  exit 2
fi

PYTHON_BIN="${RLINF_PYTHON:-${TIANJI_GPU_PYTHON:-${PYTHON_BIN:-python3}}}"
export PYTHON_BIN
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/tianji_cuda_env.sh"
"${PYTHON_BIN}" -B "${SCRIPT_DIR}/check_stage2_resume.py" --config "$@"
exec "${PYTHON_BIN}" -u "${ROOT_DIR}/examples/embodiment/train_async.py" \
  --config-name "${TIANJI_STAGE2_CONFIG:-tianji2_rlt_stage2_async}" "$@"
