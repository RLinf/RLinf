#!/usr/bin/env bash
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Source this file from the model venv after setting the run directory and SDK interpreter.
set -euo pipefail
: "${STACK_CI_RUN:?Set STACK_CI_RUN to a new CI output directory.}"
: "${STACK_CUBE_ENV_PYTHON:?Set the SDK interpreter path.}"
export EMBODICHAIN_DATA_ROOT=/workspace/dataset/embodichain_data
CUDA_VISIBLE_DEVICES="" "$STACK_CUBE_ENV_PYTHON" -m embodichain.data download --name SimResources
export STACK_CUBE_SFT_MODEL=/workspace/dataset/lerobot-pi05-base-22c30e5b
hf download lerobot/pi05_base --revision 22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0 --local-dir "$STACK_CUBE_SFT_MODEL"
hf download RLinf/IsaacLab-Stack-Cube-Data --repo-type dataset --revision 2e27f04b2e31f293fd020164917e4cce09e108d6 --local-dir "$STACK_CI_RUN/reference"
python - "$STACK_CI_RUN/reference/source_revision.json" <<'PY'
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"repo_id": "RLinf/IsaacLab-Stack-Cube-Data", "revision": "2e27f04b2e31f293fd020164917e4cce09e108d6"}))
PY
python toolkits/lerobot/prepare_stack_cube_data.py split-reference --source "$STACK_CI_RUN/reference" --output "$STACK_CI_RUN/split"
python toolkits/lerobot/calculate_stack_cube_norm_stats.py --dataset "$STACK_CI_RUN/split/train"
export STACK_CUBE_NORM_STATS="$STACK_CI_RUN/split/train/norm_stats.json"
git clone --filter=blob:none --no-checkout --depth 1 --branch v5.1.0 https://github.com/isaac-sim/IsaacSim.git "$STACK_CI_RUN/isaacsim_source"
git -C "$STACK_CI_RUN/isaacsim_source" sparse-checkout set source/extensions/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description
git -C "$STACK_CI_RUN/isaacsim_source" checkout --detach 47d886f2858d1ceed556b21c88927aa67bc81c12
python toolkits/lerobot/prepare_stack_cube_data.py prepare-panda --source "$STACK_CI_RUN/isaacsim_source/source/extensions/isaacsim.asset.importer.urdf/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf" --output "$STACK_CI_RUN/assets"
export STACK_CUBE_PANDA_URDF="$STACK_CI_RUN/assets/panda.urdf"
export STACK_CUBE_PPO_TRAIN_REQUESTS="$STACK_CI_RUN/train.json"
export STACK_CUBE_PPO_VAL_REQUESTS="$STACK_CI_RUN/val.json"
python toolkits/lerobot/prepare_stack_cube_data.py manifest --profile formal --split train --seed 700000 --episodes 32 --output "$STACK_CUBE_PPO_TRAIN_REQUESTS"
python toolkits/lerobot/prepare_stack_cube_data.py manifest --profile formal --split val --seed 710000 --episodes 8 --output "$STACK_CUBE_PPO_VAL_REQUESTS"
