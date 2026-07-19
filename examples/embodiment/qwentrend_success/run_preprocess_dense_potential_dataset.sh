#!/usr/bin/env bash
# Step 2b (dense): train state-success teacher, then build potential/progress SFT data.
set -euo pipefail

: "${UNIFORM_DATA_ROOT:?Set the root containing step0, step20, ..., step200}"
UNIFORM_STEPS=${UNIFORM_STEPS:-"0 20 40 60 80 100 120 140 160 180 200"}
: "${STATE_VALUE_ROOT:?Set the state-success value teacher output root}"
: "${POTENTIAL_SFT_DATA_ROOT:?Set the processed dense potential dataset output root}"

PYTHON_BIN=${PYTHON_BIN:-/opt/venv/openvla/bin/python}
DEVICE=${DEVICE:-cuda:0}
TEACHER_DEVICE=${TEACHER_DEVICE:-cpu}
TEACHER_BATCH_SIZE=${TEACHER_BATCH_SIZE:-4096}
TEACHER_MAX_STEPS=${TEACHER_MAX_STEPS:-3000}
FLAT_ROOT=${FLAT_ROOT:-"${PWD}/logs/qwentrend_uniform_collection_flat"}

rm -rf "${FLAT_ROOT}"
mkdir -p "${FLAT_ROOT}" "${STATE_VALUE_ROOT}" "${POTENTIAL_SFT_DATA_ROOT}"

for step in ${UNIFORM_STEPS}; do
  step_dir="${UNIFORM_DATA_ROOT}/step${step}"
  [[ -d "${step_dir}" ]] || {
    echo "missing collection dir: ${step_dir}" >&2
    exit 1
  }
  for f in "${step_dir}"/*.pkl; do
    [[ -e "$f" ]] || continue
    ln -s "$(realpath "$f")" "${FLAT_ROOT}/step${step}_$(basename "$f")"
  done
done

"${PYTHON_BIN}" examples/reward/train_state_success_value.py \
  --raw-data-path "${FLAT_ROOT}" \
  --output-dir "${STATE_VALUE_ROOT}" \
  --device "${TEACHER_DEVICE}" \
  --batch-size "${TEACHER_BATCH_SIZE}" \
  --max-steps "${TEACHER_MAX_STEPS}"

if [[ ! -f "${STATE_VALUE_ROOT}/best.pt" ]]; then
  if [[ -f "${STATE_VALUE_ROOT}/final.pt" ]]; then
    ln -sfn final.pt "${STATE_VALUE_ROOT}/best.pt"
  else
    echo "missing teacher checkpoint under ${STATE_VALUE_ROOT}" >&2
    exit 1
  fi
fi

raw_args=()
for step in ${UNIFORM_STEPS}; do
  raw_args+=(--raw-data-path "${UNIFORM_DATA_ROOT}/step${step}")
done

"${PYTHON_BIN}" examples/reward/preprocess_qwentrend_state_value_potential_dataset.py \
  "${raw_args[@]}" \
  --value-checkpoint "${STATE_VALUE_ROOT}/best.pt" \
  --output-dir "${POTENTIAL_SFT_DATA_ROOT}" \
  --device "${DEVICE}" \
  "$@"

echo "dense potential SFT data ready at ${POTENTIAL_SFT_DATA_ROOT}"
echo "state-value teacher at ${STATE_VALUE_ROOT}/best.pt"
