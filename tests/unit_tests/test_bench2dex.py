# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU contracts for the Bench2Dex evaluation protocol."""

from pathlib import Path

import pytest

from evaluations.bench2dex.run_all_channels import channel_command
from rlinf.envs.sim.bench2dex.evaluation import configure_profile, normalize_profile


@pytest.mark.parametrize(
    ("channel", "invariant", "covariant"),
    [
        ("none", False, False),
        ("cov", False, True),
        ("inv", True, False),
        ("inv_cov", True, True),
    ],
)
def test_evaluation_channels_keep_distinct_variation_axes(
    channel, invariant, covariant
):
    original = {"appearance": {"background": {"sentinel": "kept"}}}
    config = configure_profile(original, channel)
    for category, axis in (
        ("appearance", "background"),
        ("appearance", "table_surface"),
        ("appearance", "light"),
        ("clutter", "tabletop"),
        ("spatial", "camera"),
    ):
        assert config[category][axis]["enabled"] is invariant
    for axis in ("object_pose", "table_height"):
        assert config["spatial"][axis]["enabled"] is covariant
    if covariant:
        assert config["spatial"]["object_pose"]["position_jitter_cm"] == 2.0
        assert config["spatial"]["object_pose"]["yaw_jitter_deg"] == 10.0
        assert config["spatial"]["table_height"]["offset_m_range"] == [-0.05, 0.05]
    assert config["appearance"]["background"]["sentinel"] == "kept"
    assert original == {"appearance": {"background": {"sentinel": "kept"}}}


def test_evaluation_rejects_unknown_channel():
    with pytest.raises(ValueError):
        normalize_profile("unknown")


def test_parallel_evaluation_command_keeps_episode_and_seed_contract():
    command = channel_command(
        Path("/repo"),
        "example",
        "inv_cov",
        12,
        7,
        Path("/output"),
        Path("/anchors"),
        123,
        456,
        "unseen",
        [],
        num_envs=3,
    )
    assert "env.eval.rollout_epoch=4" in command
    assert "env.eval.total_num_envs=3" in command
    assert "env.eval.auto_reset=false" in command
    assert "env.eval.seed=123" in command
    assert "env.eval.init_params.inv_cov_seed=456" in command
    assert "env.eval.init_params.start_episode=7" in command
    assert "env.eval.init_params.anchor_dir=/anchors" in command
    assert "env.eval.init_params.generalization_profile=inv_cov" in command


def test_parallel_evaluation_rejects_incomplete_batch():
    with pytest.raises(ValueError):
        channel_command(
            Path("/repo"),
            "example",
            "none",
            5,
            1,
            Path("/out"),
            Path("/anchors"),
            1,
            2,
            "unseen",
            [],
            num_envs=2,
        )
