# Copyright 2026 The RLinf Authors.
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

"""Hydra composition of the four SuperMimic configs.

Skipped when Hydra cannot be imported. The schema checked here is the one
``validate_supermimic_cfg`` actually reads.
"""

from __future__ import annotations

import copy

import pytest

try:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
except Exception as exc:  # hydra-core 1.2 cannot be imported on Python 3.11
    pytest.skip(f"Hydra cannot be imported: {exc}", allow_module_level=True)

from rlinf.workers.supermimic.validate import validate_supermimic_cfg

CONFIG_DIR = (
    __import__("pathlib").Path(__file__).resolve().parents[2]
    / "examples"
    / "embodiment"
    / "config"
)
RUNNABLE = (
    "supermimic_lafan1",
    "supermimic_lafan1_anchor",
    "supermimic_smoke",
    "supermimic_lafan1_goal730",
    "supermimic_lafan1_obs99",
    "supermimic_velocity",
    "supermimic_bones_seed",
)


def compose_cfg(name: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        return compose(config_name=name)


@pytest.fixture(scope="module")
def composed():
    return {name: compose_cfg(name) for name in ("supermimic_base", *RUNNABLE)}


def test_hydra_resolves_paper_and_smoke_sizes(composed):
    lafan1 = copy.deepcopy(composed["supermimic_lafan1"])
    lafan1.env.motion_path = "/tmp/lafan1_g1"
    lafan1 = validate_supermimic_cfg(lafan1)
    assert lafan1.algorithm.k == 8 and lafan1.algorithm.H == 32
    assert lafan1.algorithm.goal_dim == 680
    assert lafan1.algorithm.obs_history_dim == 930
    assert lafan1.algorithm.critic_in_dim == 680 + 71 + 33 + 4
    assert lafan1.env.num_envs == 4096
    assert lafan1.algorithm.wm_window_batch == 2048
    assert lafan1.algorithm.imagine_batch == 4096
    assert lafan1.algorithm.total_updates == 320000
    assert lafan1.algorithm.warmup_sim_steps == 40

    anchor = copy.deepcopy(composed["supermimic_lafan1_anchor"])
    anchor.env.motion_path = "/tmp/lafan1_g1"
    anchor = validate_supermimic_cfg(anchor)
    assert anchor.algorithm.goal_anchor is True
    assert anchor.algorithm.goal_dim == 689
    assert anchor.algorithm.obs_dim == 93

    smoke = copy.deepcopy(composed["supermimic_smoke"])
    smoke.env.motion_path = "/tmp/lafan1_g1"
    smoke = validate_supermimic_cfg(smoke)
    assert smoke.env.backend == "mjlab"
    assert smoke.env.num_envs == 8
    assert smoke.algorithm.imagine_batch == 8
    assert smoke.algorithm.wm_window_batch == 64
    assert smoke.algorithm.total_updates == 3
    assert smoke.algorithm.k == 8 and smoke.algorithm.H == 32
    assert smoke.algorithm.warmup_sim_steps == 40
    assert smoke.algorithm.goal_dim == 680

    goal730 = copy.deepcopy(composed["supermimic_lafan1_goal730"])
    goal730.env.motion_path = "/tmp/lafan1_g1"
    goal730 = validate_supermimic_cfg(goal730)
    assert goal730.algorithm.goal_dim == 730
    assert goal730.algorithm.obs_dim == 93
    assert goal730.algorithm.goal_anchor is False

    obs99 = copy.deepcopy(composed["supermimic_lafan1_obs99"])
    obs99.env.motion_path = "/tmp/lafan1_g1"
    obs99 = validate_supermimic_cfg(obs99)
    assert obs99.algorithm.obs_dim == 99
    assert obs99.algorithm.obs_history_dim == 990
    assert obs99.algorithm.goal_dim == 680

    velocity = copy.deepcopy(composed["supermimic_velocity"])
    velocity.env.motion_path = "/tmp/lafan1_g1"
    velocity = validate_supermimic_cfg(velocity)
    assert velocity.algorithm.velocity_tracking is True
    assert velocity.algorithm.goal_dim == 680
    assert velocity.env.num_envs % 2 == 0

    bones = copy.deepcopy(composed["supermimic_bones_seed"])
    bones.env.motion_path = "/tmp/bones_seed"
    bones = validate_supermimic_cfg(bones)
    assert bones.env.motion_format == "bones_seed"
    assert bones.algorithm.goal_dim == 680
