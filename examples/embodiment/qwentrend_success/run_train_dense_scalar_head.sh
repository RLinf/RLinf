#!/usr/bin/env bash
# Step 3c (dense): extract frozen Qwen features, then train scalar potential head.
set -euo pipefail

: "${QWEN_MODEL_PATH:?Set the Qwen3-VL-4B-Instruct path}"
: "${POTENTIAL_SFT_DATA_ROOT:?Set the processed dense potential dataset root}"
: "${QWENTREND_POTENTIAL_CHECKPOINT:?Set the dense potential LoRA checkpoint dir}"
: "${FEAT_ROOT:?Set the feature shard output root}"
: "${SCALAR_OUTPUT_ROOT:?Set the scalar-head output root}"

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/openvla/bin/python}
FEATURE_WORLD_SIZE=${FEATURE_WORLD_SIZE:-4}
FEATURE_BATCH_SIZE=${FEATURE_BATCH_SIZE:-4}
SCALAR_EPOCHS=${SCALAR_EPOCHS:-50}
SCALAR_DEVICE=${SCALAR_DEVICE:-cuda:0}

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
      CUDA_VISIBLE_DEVICES="${rank}" "${PYTHON_BIN}" scripts/extract_qwentrend_potential_features.py \
        --model-path "${QWEN_MODEL_PATH}" \
        --checkpoint "${QWENTREND_POTENTIAL_CHECKPOINT}" \
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

"${PYTHON_BIN}" scripts/train_qwentrend_scalar_head.py \
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
