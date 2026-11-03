#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

: "${REPO_PATH:?Set REPO_PATH to the repository checkout.}"
: "${STACK_CI_RUN:?Set STACK_CI_RUN to a new CI output directory.}"
export PYTHONPATH="$REPO_PATH${PYTHONPATH:+:$PYTHONPATH}"

# This smoke checks checkpoint correctness rather than interconnect performance.
if [ "${STACK_CUBE_NCCL_SOCKET_FALLBACK:-1}" = 1 ]; then
    export NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 NCCL_IB_DISABLE=1
    export NCCL_NVLS_ENABLE=0 NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0
fi

config_dir="$REPO_PATH/tests/e2e_tests/sft"
config_name=embodichain_stack_cube_sft_openpi_pi05
python "$REPO_PATH/examples/sft/train_vla_sft.py" \
    --config-path "$config_dir" --config-name "$config_name" \
    actor.fsdp_config.save_full_model_weights=false

checkpoint="$STACK_CI_RUN/sft/ci-test/checkpoints/global_step_1"
python - "$checkpoint/actor" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
metadata = json.loads((root / "checkpoint_metadata.json").read_text())
if not metadata["complete"] or not metadata["save_training_state"]:
    raise RuntimeError("SFT stage did not publish a resumable checkpoint.")
if not (root / "openpi_loader_state.json").is_file():
    raise RuntimeError("SFT checkpoint is missing completed loader state.")
PY

python "$REPO_PATH/examples/sft/train_vla_sft.py" \
    --config-path "$config_dir" --config-name "$config_name" \
    runner.resume_dir="$checkpoint" runner.max_steps=2

python - "$STACK_CI_RUN/sft/ci-test/checkpoints/global_step_2/actor" <<'PY'
import sys
from pathlib import Path

from rlinf.hybrid_engines.fsdp.strategy.checkpoint import validate_checkpoint_for_resume
from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights

root = Path(sys.argv[1])
validate_checkpoint_for_resume(str(root))
if not (root / "openpi_loader_state.json").is_file():
    raise RuntimeError("Resumed SFT stage did not publish loader state.")
resolve_full_weights(root.parent)
PY
