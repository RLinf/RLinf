#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export EMBODIED_PATH="${EMBODIED_PATH:-$ROOT_DIR/examples/embodiment}"

DATASET_PATH="${1:-/mnt/public2/xiekaizhi/rlt-openpi-sim/tmp/rlt_maniskill_joint_wide_smoke}"
SEED="${SEED:-0}"
SIM_BACKEND="${SIM_BACKEND:-physx_cpu}"
IMAGE_WIDTH="${IMAGE_WIDTH:-384}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-384}"
export DATASET_PATH SEED SIM_BACKEND IMAGE_WIDTH IMAGE_HEIGHT

echo "[1/4] Check widened env geometry"
python - <<'PY'
import os
import sys

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

import gymnasium as gym
import mani_skill.envs  # noqa: F401

from rlinf.envs.maniskill.peg_insertion_side_variants import (
    PEG_INSERTION_SIDE_WIDE_ENV_ID,
    register_rlinf_peg_insertion_side_variants,
)

register_rlinf_peg_insertion_side_variants()

env = gym.make(
    PEG_INSERTION_SIDE_WIDE_ENV_ID,
    obs_mode="rgb",
    control_mode="pd_joint_delta_pos",
    reward_mode="sparse",
    render_mode="rgb_array",
    sim_backend=os.environ.get("SIM_BACKEND", "physx_cpu"),
    sim_config={"sim_freq": 100, "control_freq": 10},
    sensor_configs={
        "shader_pack": "default",
        "width": int(os.environ.get("IMAGE_WIDTH", "384")),
        "height": int(os.environ.get("IMAGE_HEIGHT", "384")),
    },
    max_episode_steps=100,
)
obs, _ = env.reset(seed=int(os.environ.get("SEED", "0")))
base = env.unwrapped
peg_r = float(base.peg_half_sizes[0, 1].item())
hole_r = float(base.box_hole_radii[0].item())
hole_offset_yz = base.box_hole_offsets.p[0, 1:].detach().cpu().tolist()
print("peg_radius =", peg_r)
print("hole_radius =", hole_r)
print("clearance   =", hole_r - peg_r)
print("hole_offset_yz =", hole_offset_yz)
sensors = obs.get("sensor_data", {})
print("sensor_keys =", list(sensors.keys()))
for name, payload in sensors.items():
    if isinstance(payload, dict) and "rgb" in payload:
        print("first_rgb_camera =", name)
        print("first_rgb_shape =", tuple(payload["rgb"].shape))
        break
env.close()
PY

echo "[2/4] Collect 1 joint episode"
python examples/embodiment/collect_maniskill_peg_lerobot_joint.py \
  --repo-id "$DATASET_PATH" \
  --num-episodes 1 \
  --max-attempts 50 \
  --seed "$SEED" \
  --overwrite \
  --sim-backend "$SIM_BACKEND" \
  --image-width "$IMAGE_WIDTH" \
  --image-height "$IMAGE_HEIGHT" \
  --save-videos

echo "[3/4] Check dataset fields"
python - <<'PY'
import os
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

dataset_path = os.environ["DATASET_PATH"]
ds = LeRobotDataset(dataset_path, download_videos=False)
row = ds[0]
print("keys =", sorted(row.keys()))
print("state_shape =", np.asarray(row["state"]).shape)
print("action_shape =", np.asarray(row["actions"]).shape)
print("image_shape =", np.asarray(row["image"]).shape)
print("wrist_image_shape =", np.asarray(row["wrist_image"]).shape)
print("task =", row["task"])
PY

echo "[4/4] Replay episode 0"
python examples/embodiment/replay_maniskill_peg_lerobot_joint.py \
  --dataset-path "$DATASET_PATH" \
  --episode-index 0 \
  --sim-backend "$SIM_BACKEND" \
  --image-width "$IMAGE_WIDTH" \
  --image-height "$IMAGE_HEIGHT"

echo "Smoke test completed."
