# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smoke the RoboDojo environment backend end to end.

Boots the environment through RLinf's registry with the same settings the
``env/robodojo_put_bottles_into_dustbin`` config uses, resets one slot, applies
a few zero actions and checks the observation and step contract. Requires the
``rlinf-robodojo-runtime`` distribution and the RoboDojo scene assets, which the
embodied e2e workflow installs and mounts.

Run from the repository root::

    python tests/e2e_tests/embodied/robodojo_env_smoke.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.envs import get_env_cls  # noqa: E402

ACTION_DIM = 14


def build_cfg() -> OmegaConf:
    assets_root = os.environ.get("ROBODOJO_ASSETS_ROOT", "/workspace/dataset/robodojo")
    return OmegaConf.create(
        {
            "env_type": "robodojo",
            "seed": 0,
            "group_size": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": True,
            "use_custom_reward": True,
            "reward_coef": 1.0,
            "center_crop": False,
            "use_fixed_reset_state_ids": True,
            "max_episode_steps": 50,
            "assets_path": assets_root,
            "task_config": {
                "task_name": os.environ.get(
                    "ROBODOJO_TASK", "put_bottles_into_dustbin"
                ),
                "env_cfg_type": os.environ.get("ROBODOJO_ENV_CFG_TYPE", "arx_x5"),
                "headless": True,
                "cuda_device": int(os.environ.get("ROBODOJO_DEVICE_ID", "0")),
                "max_episode_steps": 50,
                "save_dir": os.environ.get(
                    "ROBODOJO_SMOKE_OUTPUT", "/tmp/robodojo-smoke"
                ),
            },
        }
    )


def main() -> int:
    steps = int(os.environ.get("ROBODOJO_SMOKE_STEPS", "3"))
    layout = int(os.environ.get("ROBODOJO_LAYOUT", "0"))
    env = get_env_cls("robodojo")(build_cfg(), 1, 0, 1, None, record_metrics=True)
    try:
        obs, infos = env.reset(env_seeds=[layout])
        assert infos == {}, infos
        assert obs["main_images"].shape[0] == 1, obs["main_images"].shape
        assert obs["states"].shape == (1, ACTION_DIM), obs["states"].shape
        assert len(obs["task_descriptions"]) == 1
        print(f"reset ok: images={tuple(obs['main_images'].shape)}", flush=True)

        for step in range(steps):
            obs, reward, terminated, truncated, infos = env.step(
                np.zeros((1, ACTION_DIM), dtype=np.float32)
            )
            assert reward.shape == (1,), reward.shape
            assert terminated.shape == (1,) and truncated.shape == (1,)
            assert env.elapsed_steps.tolist() == [step + 1], env.elapsed_steps
            print(
                f"step {step + 1} ok: reward={float(reward[0]):.3f} "
                f"terminated={bool(terminated[0])} truncated={bool(truncated[0])}",
                flush=True,
            )
    finally:
        env.offload()
    print("robodojo env smoke passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
