# Copyright 2026 Shirui Chen
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

"""Tests for staged YAM VLM planner GPU placement selection."""

from omegaconf import OmegaConf

from examples.embodiment.train_embodied_agent_staged import (
    _compute_vlm_gpu_index,
    _get_vlm_planner_placement,
)


def _make_cfg(
    actor_placement: int = 0,
    rollout_placement: int = 1,
    env_placement: int = 0,
    vlm_placement: int | None = None,
    vlm_node_group: str | None = None,
):
    vlm_planner = {}
    if vlm_placement is not None:
        vlm_planner["placement"] = vlm_placement
    if vlm_node_group is not None:
        vlm_planner["node_group"] = vlm_node_group

    return OmegaConf.create(
        {
            "cluster": {
                "component_placement": {
                    "actor": {"node_group": "gpu", "placement": actor_placement},
                    "rollout": {"node_group": "gpu", "placement": rollout_placement},
                    "env": {"node_group": "gpu", "placement": env_placement},
                },
                "node_groups": [
                    {"label": "gpu", "node_ranks": 0},
                    {"label": "beaker_vlm", "node_ranks": 0},
                ],
            },
            "vlm_planner": vlm_planner,
        }
    )


def test_vlm_planner_defaults_to_next_free_gpu_on_shared_node():
    cfg = _make_cfg(actor_placement=0, rollout_placement=1, env_placement=0)

    assert _compute_vlm_gpu_index(cfg) == 2
    assert _get_vlm_planner_placement(cfg) == ("beaker_vlm", 2)


def test_vlm_planner_defaults_to_gpu_zero_when_no_second_gpu_is_claimed():
    cfg = _make_cfg(actor_placement=0, rollout_placement=0, env_placement=0)

    assert _compute_vlm_gpu_index(cfg) == 0
    assert _get_vlm_planner_placement(cfg) == ("beaker_vlm", 0)


def test_vlm_planner_honors_explicit_override():
    cfg = _make_cfg(vlm_placement=5, vlm_node_group="custom_vlm")

    assert _compute_vlm_gpu_index(cfg) == 5
    assert _get_vlm_planner_placement(cfg) == ("custom_vlm", 5)
