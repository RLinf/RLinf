#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${REPO_PATH}/.venv/bin/python"
FASTWAM_PATH="${FASTWAM_PATH:-${REPO_PATH}/.venv/FastWAM}"
CHECKPOINT_DIR="${FASTWAM_CHECKPOINT_DIR:-${REPO_PATH}/checkpoints/fastwam_release}"
MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${REPO_PATH}/checkpoints}"
DATASET_ROOT="${FASTWAM_DATASET_ROOT:-${REPO_PATH}/data/libero_mujoco3.3.2}"
if [[ -n "${FASTWAM_DATASET_DIR:-}" ]]; then
    # Keep a one-suite override for smoke/debug runs.
    DATASET_DIRS=("${FASTWAM_DATASET_DIR}")
else
    # Match the official task/libero_uncond_2cam224_1e-4.yaml.
    DATASET_DIRS=(
        "${DATASET_ROOT}/libero_spatial_no_noops_lerobot"
        "${DATASET_ROOT}/libero_object_no_noops_lerobot"
        "${DATASET_ROOT}/libero_goal_no_noops_lerobot"
        "${DATASET_ROOT}/libero_10_no_noops_lerobot"
    )
fi
DATASET_DIR="${DATASET_DIRS[0]}"
TEXT_CACHE_DIR="${FASTWAM_TEXT_EMBEDDING_CACHE_DIR:-${DATASET_ROOT}/text_embeds_cache/libero}"
DATASET_DIRS_HYDRA=$(IFS=,; echo "${DATASET_DIRS[*]}")
ACTION_DIT_BACKBONE_PATH="${FASTWAM_ACTION_DIT_BACKBONE_PATH:-${MODEL_BASE_PATH}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
DOWNLOAD_DATA="${FASTWAM_DOWNLOAD_DATA:-1}"

export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"

if command -v hf >/dev/null 2>&1; then
    HF_BIN=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_BIN=huggingface-cli
else
    echo "The Hugging Face CLI is required. Activate .venv first." >&2
    exit 1
fi
if [ ! -x "${PYTHON_BIN}" ]; then
    echo "RLinf venv not found at ${PYTHON_BIN}; run requirements/install.sh first." >&2
    exit 1
fi

mkdir -p "${CHECKPOINT_DIR}" "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B" \
    "${MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B" "${DATASET_ROOT}"

echo "[FastWAM] Downloading the released LIBERO checkpoint and stats..."
"${HF_BIN}" download yuanty/fastwam \
    libero_uncond_2cam224.pt \
    libero_uncond_2cam224_dataset_stats.json \
    --local-dir "${CHECKPOINT_DIR}"

echo "[FastWAM] Downloading the VAE and T5 weights used by RLinf..."
"${HF_BIN}" download Wan-AI/Wan2.2-TI2V-5B \
    Wan2.2_VAE.pth \
    models_t5_umt5-xxl-enc-bf16.pth \
    --local-dir "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B"
"${HF_BIN}" download Wan-AI/Wan2.1-T2V-1.3B \
    --include "google/umt5-xxl/*" \
    --local-dir "${MODEL_BASE_PATH}/Wan-AI/Wan2.1-T2V-1.3B"

echo "[FastWAM] Downloading the official Wan2.2 video DiT for base-model SFT..."
if ! find "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B" -maxdepth 1 -type f \
    -name 'diffusion_pytorch_model*.safetensors' -print -quit | grep -q .; then
    "${HF_BIN}" download Wan-AI/Wan2.2-TI2V-5B \
        --include "diffusion_pytorch_model*.safetensors" \
        --local-dir "${MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B"
fi

if [ ! -s "${ACTION_DIT_BACKBONE_PATH}" ]; then
    echo "[FastWAM] Preprocessing the official ActionDiT backbone..."
    mkdir -p "$(dirname -- "${ACTION_DIT_BACKBONE_PATH}")"
    "${PYTHON_BIN}" "${FASTWAM_PATH}/scripts/preprocess_action_dit_backbone.py" \
        --model-config "${FASTWAM_PATH}/configs/model/fastwam.yaml" \
        --output "${ACTION_DIT_BACKBONE_PATH}" \
        --device "${FASTWAM_ACTION_DIT_DEVICE:-cuda}" \
        --dtype bfloat16
fi

if [ "${DOWNLOAD_DATA}" = "1" ]; then
    echo "[FastWAM] Downloading the official LIBERO LeRobot archive set..."
    "${HF_BIN}" download --repo-type dataset yuanty/LIBERO-fastwam \
        --local-dir "${DATASET_ROOT}"
    shopt -s nullglob
    archives=("${DATASET_ROOT}"/*.tar.gz)
    if [ "${#archives[@]}" -eq 0 ]; then
        echo "No LIBERO tar.gz archives found in ${DATASET_ROOT}." >&2
        exit 1
    fi
    for archive in "${archives[@]}"; do
        echo "[FastWAM] Extracting ${archive}"
        tar -xzf "${archive}" -C "${DATASET_ROOT}"
    done
fi

for dataset_dir in "${DATASET_DIRS[@]}"; do
    if [ ! -f "${dataset_dir}/meta/tasks.jsonl" ]; then
        echo "Expected LIBERO dataset at ${dataset_dir} was not found." >&2
        echo "Extract the official archive set or set FASTWAM_DATASET_DIR for a one-suite run." >&2
        exit 1
    fi
done

mkdir -p "${TEXT_CACHE_DIR}"
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"
echo "[FastWAM] Precomputing T5 text embeddings..."
"${PYTHON_BIN}" "${FASTWAM_PATH}/scripts/precompute_text_embeds.py" \
    task=libero_uncond_2cam224_1e-4 \
    "data.train.dataset_dirs=[${DATASET_DIRS_HYDRA}]" \
    "data.train.text_embedding_cache_dir=${TEXT_CACHE_DIR}" \
    +overwrite=false \
    model.redirect_common_files=true

cat <<EOF
FastWAM SFT assets are ready.
  FASTWAM_CHECKPOINT_DIR=${CHECKPOINT_DIR}
  DIFFSYNTH_MODEL_BASE_PATH=${MODEL_BASE_PATH}
  FASTWAM_DATASET_ROOT=${DATASET_ROOT}
  FASTWAM_DATASET_DIRS=${DATASET_DIRS_HYDRA}
  FASTWAM_DATASET_DIR=${DATASET_DIR}
  FASTWAM_TEXT_EMBEDDING_CACHE_DIR=${TEXT_CACHE_DIR}
  FASTWAM_ACTION_DIT_BACKBONE_PATH=${ACTION_DIT_BACKBONE_PATH}
EOF
