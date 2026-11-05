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

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco_warp", reason="mujoco-warp is not installed")
pytest.importorskip("warp", reason="warp-lang is not installed")

import mujoco

from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs.sim.mujoco_warp.mujoco_warp_env import TASK_SPECS, MujocoWarpEnv
from rlinf.envs.wrappers.record_video import RecordVideo


class _Cfg:
    def __init__(self, task_name="half_cheetah", **overrides):
        self.task_name = task_name
        self.seed = 0
        self.auto_reset = True
        self.max_episode_steps = 1000
        self.group_size = 1
        self.device = "cpu"
        for key, value in overrides.items():
            setattr(self, key, value)


def _make_env(task_name="half_cheetah", num_envs=4, **overrides):
    return MujocoWarpEnv(
        _Cfg(task_name, **overrides),
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def test_env_registration():
    env_cls = get_env_cls(SupportedEnvType.MUJOCO_WARP.value)
    assert env_cls is MujocoWarpEnv


def test_all_tasks_reset_and_step_shapes():
    for task_name in TASK_SPECS.keys():
        env = _make_env(task_name)
        obs, infos = env.reset()
        assert obs["states"].shape == (env.num_envs, env.obs_dim)
        actions = torch.rand(env.num_envs, env.action_dim) * 2 - 1
        obs, rewards, terminations, truncations, infos = env.step(actions)
        assert obs["states"].shape == (env.num_envs, env.obs_dim)
        assert rewards.shape == (env.num_envs,)
        assert terminations.dtype == torch.bool
        assert truncations.dtype == torch.bool
        assert torch.isfinite(obs["states"]).all()


def test_truncation_triggers_auto_reset_and_episode_metrics():
    env = _make_env(max_episode_steps=3)
    env.reset()
    actions = torch.zeros(env.num_envs, env.action_dim)
    for _ in range(3):
        obs, rewards, terminations, truncations, infos = env.step(actions)
    assert truncations.all()
    assert "final_observation" in infos
    assert infos["final_info"]["episode"]["episode_len"].max().item() == 3
    # After auto-reset the returned observation is finite and elapsed
    # steps restart from zero.
    assert torch.isfinite(obs["states"]).all()
    assert env.elapsed_steps.max().item() == 0


def test_chunk_step_aggregates_dones_to_last_step():
    env = _make_env(max_episode_steps=2)
    env.reset()
    actions = torch.zeros(env.num_envs, 3, env.action_dim)
    obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(actions)
    assert len(obs_list) == 3
    assert rewards.shape == (env.num_envs, 3)
    # With auto_reset, the chunk-level done flags land on the last sub-step.
    assert truncations[:, -1].all()
    assert not truncations[:, :-1].any()
    assert infos_list[1]["final_info"]["episode"]["episode_len"].eq(2).all()
    assert infos_list[-1]["final_info"]["episode"]["episode_len"].eq(2).all()
    assert torch.isfinite(infos_list[-1]["final_observation"]["states"]).all()


def test_reset_with_same_state_ids_is_deterministic_per_world():
    env = _make_env(num_envs=2)
    obs1, _ = env.reset(reset_state_ids=np.array([1, 2]))
    obs2, _ = env.reset(reset_state_ids=np.array([1, 2]))
    assert torch.allclose(obs1["states"], obs2["states"])

    permuted, _ = env.reset(reset_state_ids=np.array([2, 1]))
    assert torch.allclose(obs1["states"][0], permuted["states"][1])
    assert torch.allclose(obs1["states"][1], permuted["states"][0])

    colliding_sum, _ = env.reset(reset_state_ids=np.array([0, 3]))
    assert not torch.allclose(obs1["states"], colliding_sum["states"])


def test_initial_partial_reset_initializes_all_worlds():
    env = _make_env(num_envs=2, specific_reset_id=0)
    partial, _ = env.reset(env_idx=[1], reset_state_ids=np.array([0]))
    reference, _ = env.reset(reset_state_ids=np.array([0, 0]))
    assert torch.allclose(partial["states"], reference["states"])


def test_reward_scale_preserves_dynamics_and_scales_rewards():
    base_env = _make_env(num_envs=2)
    scaled_env = _make_env(num_envs=2, reward_scale=2.5)
    reset_ids = np.array([3, 7])
    base_env.reset(reset_state_ids=reset_ids)
    scaled_env.reset(reset_state_ids=reset_ids)
    actions = torch.full((2, base_env.action_dim), 0.25)

    base_obs, base_rewards, _, _, _ = base_env.step(actions)
    scaled_obs, scaled_rewards, _, _, _ = scaled_env.step(actions)

    assert torch.allclose(scaled_obs["states"], base_obs["states"])
    assert torch.allclose(scaled_rewards, base_rewards * 2.5)


def test_video_frames_follow_worlds_and_staggered_terminal_states(monkeypatch):
    class FakeRenderer:
        def __init__(self, model, height, width):
            self.height = height
            self.width = width
            self.value = 0.0
            self.closed = False

        def update_scene(self, data, camera):
            assert camera.trackbodyid == 1
            self.value = float(data.qpos[1])

        def render(self):
            packed = np.array([self.value], dtype=np.float32).view(np.uint8)
            frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
            for channel in range(3):
                frame[..., channel] = packed[channel]
            return frame

        def close(self):
            self.closed = True

    monkeypatch.setattr(mujoco, "Renderer", FakeRenderer)
    video_cfg = {"save_video": True, "info_on_video": False}
    env = _make_env(
        num_envs=2,
        max_episode_steps=2,
        video_cfg=video_cfg,
        render_width=12,
        render_height=8,
    )
    wrapped = RecordVideo(env, video_cfg)
    obs, _ = wrapped.reset(reset_state_ids=np.array([0, 1]))
    assert obs["images"].shape == (2, 8, 12, 3)
    assert obs["images"].dtype == np.uint8
    assert not np.array_equal(obs["images"][0], obs["images"][1])
    assert len(wrapped.render_images) == 1

    env._elapsed_steps.copy_(torch.tensor([1, 0]))
    result = wrapped.chunk_step(torch.zeros(2, 2, env.action_dim))
    _, _, _, truncations, infos_list = result
    assert truncations[:, -1].all()
    first_final_frames = infos_list[0]["final_observation"]["images"]
    final_frames = infos_list[-1]["final_observation"]["images"]
    assert np.array_equal(final_frames[0], first_final_frames[0])
    assert not np.array_equal(final_frames[1], first_final_frames[1])
    assert len(wrapped.render_images) > 1

    renderer = env._renderer
    wrapped.close()
    assert renderer.closed
    assert env._renderer is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
