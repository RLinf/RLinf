#!/usr/bin/env bash
# Canonical launcher for Pi0.5 full-parameter ALOHA SFT on 2x H100 80GB.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "${SCRIPT_DIR}/../.." && pwd)
VENV="${VENV:-${REPO}/.venv}"

# All paths can be overridden from the environment without editing this file.
export ALOHA_DATASET_ROOT="${ALOHA_DATASET_ROOT:-${REPO}/data/lerobot-data_mixed_8_v30}"
export ALOHA_NORM_STATS_PATH="${ALOHA_NORM_STATS_PATH:-${ALOHA_DATASET_ROOT}/norm_stats.json}"
export PI05_MODEL_PATH="${PI05_MODEL_PATH:-/inspire/hdd/global_user/czxs253130583/fangchuan/data/model/lerobot/pi05_base}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$(dirname -- "${ALOHA_DATASET_ROOT}")}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export EMBODIED_PATH="${REPO}/examples/sft"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [ ! -f "${VENV}/bin/activate" ]; then
    echo "[launcher] ERROR: Python environment not found: ${VENV}" >&2
    exit 1
fi
# The repository venv contains RLinf/OpenPI dependencies. Activating an outer
# Conda environment is neither required nor sufficient for this launcher.
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${REPO}"

for required_path in \
    "${ALOHA_DATASET_ROOT}/meta/info.json" \
    "${ALOHA_NORM_STATS_PATH}" \
    "${PI05_MODEL_PATH}/model.safetensors"; do
    if [ ! -f "${required_path}" ]; then
        echo "[launcher] ERROR: required file not found: ${required_path}" >&2
        exit 1
    fi
done

if [ "${SKIP_GPU_CHECK:-0}" != "1" ]; then
    GPU_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())')
    if [ "${GPU_COUNT}" -lt 2 ]; then
        echo "[launcher] ERROR: this FSDP config requires 2 visible GPUs; found ${GPU_COUNT}." >&2
        echo "[launcher] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" >&2
        echo "[launcher] Run this launcher on the intended 2-GPU node." >&2
        exit 1
    fi
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-/inspire/hdd/global_user/czxs253130583/fangchuan/work/RL/output/model/sft/pi05_rlinf}"
EXPERIMENT_NAME=aloha_sft_openpi_pi05_full_fsdp2h100

# ---------------------------------------------------------------------------
# 断点续训: 默认自动寻找 OUTPUT_ROOT 下最新保存的 global_step_* checkpoint
# 并恢复 model + optimizer + lr_scheduler 完整训练状态。
# 覆盖方式:
#   RESUME_DIR=/path/to/checkpoints/global_step_XXXX  -> 显式指定恢复点
#   RESUME=0                                          -> 从头训练 (不恢复)
# ---------------------------------------------------------------------------
RESUME_DIR="${RESUME_DIR:-}"
if [ "${RESUME:-1}" = "0" ]; then
    RESUME_DIR=""
else
    if [ -n "${RESUME_DIR}" ] && [ ! -d "${RESUME_DIR}/actor" ]; then
        echo "[launcher] ERROR: RESUME_DIR=${RESUME_DIR} 下不存在 actor 子目录" >&2
        exit 1
    fi
    if [ -z "${RESUME_DIR}" ]; then
        RESUME_DIR=$(find "${OUTPUT_ROOT}" -maxdepth 4 -type d -name 'global_step_*' \
            -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)
    fi
fi
if [ -n "${RESUME_DIR}" ]; then
    # 沿用该 checkpoint 所属 run 的目录, 使 tensorboard / checkpoint 连续续写
    RUN_ID="${RUN_ID:-$(basename "$(dirname "$(dirname "$(dirname "${RESUME_DIR}")")")")}"
    echo "[launcher] 断点续训: resume_dir=${RESUME_DIR}"
else
    echo "[launcher] 未找到可恢复的 checkpoint, 从头开始训练"
fi
RUN_ID="${RUN_ID:-$(date +'%Y%m%d-%H%M%S')-aloha-openpi-rlinf-full-fsdp2h100}"
LOG_DIR="${OUTPUT_ROOT}/${RUN_ID}"
mkdir -p "${LOG_DIR}"

echo "Launching full-parameter Pi0.5 SFT on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "Dataset: ${ALOHA_DATASET_ROOT}"
echo "Norm stats: ${ALOHA_NORM_STATS_PATH}"
echo "Base model: ${PI05_MODEL_PATH}"
echo "Output: ${LOG_DIR}"
TRAIN_CMD=(
    python examples/sft/train_vla_sft.py
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${EXPERIMENT_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
)
if [ -n "${RESUME_DIR}" ]; then
    TRAIN_CMD+=("runner.resume_dir=${RESUME_DIR}")
fi
TRAIN_CMD+=("$@")
exec "${TRAIN_CMD[@]}"
