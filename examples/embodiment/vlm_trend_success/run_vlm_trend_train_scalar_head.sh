#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# Extract frozen VLM features, then train the scalar potential head.
set -euo pipefail

: "${VLM_MODEL_PATH:?Set the VLM base model path (e.g. Qwen3-VL-4B-Instruct)}"
: "${POTENTIAL_SFT_DATA_ROOT:?Set the processed dense potential dataset root}"
: "${VLM_TREND_POTENTIAL_CHECKPOINT:?Set the dense potential LoRA checkpoint dir}"
: "${FEAT_ROOT:?Set the feature shard output root}"
: "${SCALAR_OUTPUT_ROOT:?Set the scalar-head output root}"

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/openvla/bin/python}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
IFS=',' read -r -a FEATURE_GPUS <<< "${CUDA_DEVICES}"
FEATURE_WORLD_SIZE=${FEATURE_WORLD_SIZE:-${#FEATURE_GPUS[@]}}
FEATURE_BATCH_SIZE=${FEATURE_BATCH_SIZE:-4}
SCALAR_EPOCHS=${SCALAR_EPOCHS:-50}
SCALAR_DEVICE=${SCALAR_DEVICE:-cuda:0}

if ((FEATURE_WORLD_SIZE < 1 || FEATURE_WORLD_SIZE > ${#FEATURE_GPUS[@]})); then
  echo "FEATURE_WORLD_SIZE must be between 1 and ${#FEATURE_GPUS[@]}" >&2
  exit 1
fi

mkdir -p "${FEAT_ROOT}" "${SCALAR_OUTPUT_ROOT}"

for split in train eval; do
  manifest="${POTENTIAL_SFT_DATA_ROOT}/${split}/segments.jsonl"
  [[ -f "${manifest}" ]] || {
    echo "missing manifest: ${manifest}" >&2
    exit 1
  }
  for sample_type in potential progress; do
    pids=()
    for rank in $(seq 0 $((FEATURE_WORLD_SIZE - 1))); do
      gpu="${FEATURE_GPUS[$rank]}"
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/extract_vlm_trend_potential_features.py \
        --model-path "${VLM_MODEL_PATH}" \
        --checkpoint "${VLM_TREND_POTENTIAL_CHECKPOINT}" \
        --manifest "${manifest}" \
        --output "${FEAT_ROOT}/${split}_${sample_type}_rank${rank}.pt" \
        --sample-type "${sample_type}" \
        --device cuda:0 \
        --batch-size "${FEATURE_BATCH_SIZE}" \
        --rank "${rank}" \
        --world-size "${FEATURE_WORLD_SIZE}" &
      pids+=($!)
    done
    for pid in "${pids[@]}"; do
      wait "${pid}"
    done
  done
done

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" "${PYTHON_BIN}" scripts/train_vlm_trend_scalar_head.py \
  --train-pattern "${FEAT_ROOT}/train_potential_rank*.pt" \
  --eval-pattern "${FEAT_ROOT}/eval_potential_rank*.pt" \
  --progress-pattern "${FEAT_ROOT}/eval_progress_rank*.pt" \
  --train-progress-pattern "${FEAT_ROOT}/train_progress_rank*.pt" \
  --output-dir "${SCALAR_OUTPUT_ROOT}" \
  --device "${SCALAR_DEVICE}" \
  --epochs "${SCALAR_EPOCHS}" \
  "$@"

[[ -f "${SCALAR_OUTPUT_ROOT}/best.pt" ]] || {
  echo "missing ${SCALAR_OUTPUT_ROOT}/best.pt" >&2
  exit 1
}
echo "dense scalar head ready at ${SCALAR_OUTPUT_ROOT}/best.pt"
