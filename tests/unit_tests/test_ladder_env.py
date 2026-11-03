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

"""Unit tests for the LADDER environment integration."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.sim.ladder import (
    ROBOLAB_TASKS,
    ROBOT_ARENA_TASKS,
    SIMPLER_TASKS,
    SUITE_TASKS,
    LadderEnv,
)


def test_supported_env_type_and_catalogs() -> None:
    assert SupportedEnvType("ladder") == SupportedEnvType.LADDER
    assert get_env_cls("ladder") is LadderEnv
    assert len(SIMPLER_TASKS) == 54
    assert len(ROBOT_ARENA_TASKS) == 47
    assert len(ROBOLAB_TASKS) == 44
    assert sum(len(v) for v in SUITE_TASKS.values()) == 145


def test_prepare_actions_for_ladder_widowx() -> None:
    raw = np.zeros((2, 4, 7), dtype=np.float32)
    raw[..., -1] = np.array(
        [[0.8, 0.2, 0.9, 0.1], [0.1, 0.9, 0.2, 0.8]], dtype=np.float32
    )
    out = prepare_actions(
        raw_chunk_actions=raw,
        env_type="ladder",
        model_type="openvla",
        num_action_chunks=4,
        action_dim=7,
        action_scale=1.0,
        policy="widowx_bridge",
    )
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4, 7)
    assert torch.allclose(out[0, :, -1], torch.tensor([1.0, -1.0, 1.0, -1.0]))


def test_ladder_env_reset_step_and_chunk_step_with_mock_sim() -> None:
    cfg = OmegaConf.create(
        {
            "env_type": "ladder",
            "seed": 0,
            "group_size": 1,
            "auto_reset": False,
            "use_rel_reward": True,
            "ignore_terminations": False,
            "use_fixed_reset_state_ids": False,
            "reward_coef": 1.0,
            "reward_mode": "subgoal_progress",
            "max_episode_steps": 10,
            "init_params": {
                "suite": "simpler",
                "task_name": "all2_plate",
                "prompt_mode": "oracle_hlc",
                "camera_height": 64,
                "camera_width": 64,
            },
        }
    )

    def _fake_obs(done: bool, achieved: list[bool]) -> SimpleNamespace:
        return SimpleNamespace(
            images={"main": np.zeros((64, 64, 3), dtype=np.uint8)},
            state=SimpleNamespace(
                eef_pose=np.eye(4, dtype=np.float32),
                eef_pose_from_target=np.eye(4, dtype=np.float32),
                gripper_state=np.array([1.0], dtype=np.float32),
                joints=np.zeros(7, dtype=np.float32),
            ),
            metadata={
                "instruction": "put carrot and spoon on plate",
                "done": done,
                "truncated": False,
                "oracle": {
                    "subtasks": ["put carrot on plate", "put spoon on plate"],
                    "achieved": achieved,
                },
            },
        )

    with (
        patch.object(
            LadderEnv,
            "_reset_backend",
            return_value=[
                _fake_obs(False, [False, False]),
                _fake_obs(False, [False, False]),
            ],
        ),
        patch.object(LadderEnv, "_decode_action", side_effect=lambda env_i, raw: raw),
        patch.object(
            LadderEnv,
            "_step_backend",
            return_value=[
                _fake_obs(True, [True, True]),
                _fake_obs(False, [True, False]),
            ],
        ),
    ):
        env = LadderEnv(cfg, num_envs=2, seed_offset=0, total_num_processes=1)
        obs, _ = env.reset()
        assert obs["main_images"].shape == (2, 64, 64, 3)
        assert obs["states"].shape == (2, 7)
        assert obs["task_descriptions"] == [
            "put carrot on plate",
            "put carrot on plate",
        ]

        chunk = torch.zeros((2, 2, 7), dtype=torch.float32)
        obs_list, rewards, terms, truncs, infos_list = env.chunk_step(chunk)
        assert len(obs_list) == 2
        assert rewards.shape == (2, 2)
        assert terms.shape == (2, 2)
        assert truncs.shape == (2, 2)
        assert torch.allclose(rewards[:, 0].cpu(), torch.tensor([1.0, 0.5]))
        assert torch.allclose(rewards[:, 1].cpu(), torch.tensor([0.0, 0.0]))
        assert torch.equal(
            infos_list[-1]["episode"]["success_once"].cpu(), torch.tensor([True, False])
        )
        assert torch.allclose(
            infos_list[-1]["episode"]["subgoal_progress"].cpu(),
            torch.tensor([1.0, 0.5]),
        )
        assert obs_list[-1]["task_descriptions"] == [
            "put spoon on plate",
            "put spoon on plate",
        ]
        env.close()


def test_ladder_invalid_suite_or_task_raises() -> None:
    bad_suite_cfg = OmegaConf.create({"init_params": {"suite": "invalid"}})
    with pytest.raises(ValueError, match="Unknown LADDER suite"):
        LadderEnv(bad_suite_cfg, num_envs=1, seed_offset=0, total_num_processes=1)

    bad_task_cfg = OmegaConf.create(
        {"init_params": {"suite": "simpler", "task_name": "not_a_task"}}
    )
    with pytest.raises(ValueError, match="Unknown task"):
        LadderEnv(bad_task_cfg, num_envs=1, seed_offset=0, total_num_processes=1)
