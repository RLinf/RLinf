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

"""World-model env episode handling and the BWM backend."""

from __future__ import annotations

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.world_model import registry


class _ConstantRewardModel(torch.nn.Module):
    def __init__(self, reward: float):
        super().__init__()
        self.reward = reward

    def predict_rew(self, frames):
        return torch.full((frames.shape[0],), self.reward, device=frames.device)


class _BlankFrameBackend:
    """World-model backend that draws blank frames; the reward model scores a constant."""

    supports_kir = False
    condition_frame_length = 2
    image_size = (8, 8)

    def __init__(self, cfg, device):
        self.chunk = cfg.chunk

    @staticmethod
    def load_reward_model(cfg):
        return _ConstantRewardModel(cfg.unit_test_reward)

    def reward_instructions(self, env):
        return None

    def open_session(self, env_ids, init_frames, init_actions, seeds):
        pass

    def generate(self, env_ids, actions):
        return torch.zeros(len(env_ids), 3, self.chunk, *self.image_size)

    def close_session(self, env_ids):
        pass

    def offload(self):
        pass

    def onload(self):
        pass


def _blank_world_model_env(tmp_path, monkeypatch, *, reward, **overrides):
    from rlinf.envs.sim.world_model.env import WorldModelEnv

    monkeypatch.setitem(registry._REGISTRY, "unit_test_blank", _BlankFrameBackend)
    frame = {
        "image": np.zeros((8, 8, 3), dtype=np.uint8),
        "task": "adjust the bottle",
        "delta_action": np.zeros(14, dtype=np.float32),
    }
    for episode in range(2):
        np.save(tmp_path / f"episode_{episode}.npy", np.array([frame, frame]))
    cfg = OmegaConf.create(
        {
            "backend": "unit_test_blank",
            "wm_env_type": "robotwin",
            "seed": 0,
            "auto_reset": True,
            "ignore_terminations": True,
            "max_episode_steps": 4,
            "chunk": 2,
            "action_dim": 14,
            "reward_coef": 1.0,
            "unit_test_reward": reward,
            "initial_image_path": str(tmp_path),
            "enable_kir": False,
            "use_fixed_reset_state_ids": True,
            "group_size": 1,
            "video_cfg": {"save_video": False},
            **overrides,
        }
    )
    env = WorldModelEnv(cfg, num_envs=2, seed_offset=0, total_num_processes=1)
    env.reset()
    return env


class _ScriptedRewardModel(torch.nn.Module):
    """Scores every frame of the n-th scored chunk with ``scores[n]``."""

    def __init__(self, scores):
        super().__init__()
        self.scores = list(scores)
        self.calls = 0

    def predict_rew(self, frames):
        self.calls += 1
        return torch.full((frames.shape[0],), self.scores.pop(0), device=frames.device)


class _ScriptedRewardBackend(_BlankFrameBackend):
    @staticmethod
    def load_reward_model(cfg):
        return _ScriptedRewardModel(cfg.unit_test_scores)


def test_world_model_env_episode_reward_pays_the_window_peak_at_the_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setitem(
        registry._REGISTRY, "unit_test_scripted", _ScriptedRewardBackend
    )
    # Episodes are two chunks of two frames; frames 1-2 precede reward_first_frame,
    # so the first chunk of each episode is not scored at all.
    env = _blank_world_model_env(
        tmp_path,
        monkeypatch,
        reward=0.0,
        backend="unit_test_scripted",
        unit_test_scores=[1.0, 0.5],
        reward_mode="episode_window_max",
        reward_first_frame=3,
        terminate_at_max_episode_steps=True,
    )
    actions = torch.zeros(2, 2, 14)

    rewards = [env.chunk_step(actions)[1] for _ in range(4)]

    assert env.reward_model.calls == 2
    assert not rewards[0].any()
    assert rewards[1][:, -1].eq(1).all() and not rewards[1][:, :-1].any()
    # The previous episode's peak does not carry over.
    assert not rewards[2].any() and not rewards[3].any()


def test_world_model_env_echoes_the_last_command_as_state(tmp_path, monkeypatch):
    rest = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
    env = _blank_world_model_env(
        tmp_path, monkeypatch, reward=0.0, state_from_last_action=True, rest_action=rest
    )
    rest_states = torch.tensor([rest, rest])
    actions = torch.arange(2 * 2 * 14, dtype=torch.float32).reshape(2, 2, 14)

    obs, _ = env.reset()
    assert torch.equal(obs["states"].cpu(), rest_states)

    obs_list, *_ = env.chunk_step(actions)
    assert torch.equal(obs_list[0]["states"].cpu(), actions[:, -1])

    # The episode limit restarts both slots at the rest pose.
    obs_list, _, _, _, infos = env.chunk_step(actions + 1)
    assert torch.equal(obs_list[0]["states"].cpu(), rest_states)
    assert torch.equal(
        infos[0]["final_observation"]["states"].cpu(), actions[:, -1] + 1
    )


class _RecordingBwmPipeline:
    torch_dtype = torch.float32

    def __init__(self, image_size):
        self.image_size = image_size
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return torch.full((1, 3, 61, *self.image_size), float(len(self.calls)))


class _HalfActionBridge:
    def __init__(self, **kwargs):
        pass

    def __call__(self, joint14):
        return np.asarray(joint14, dtype=np.float32) * 0.5


def test_bwm_backend_rolls_its_condition_window(monkeypatch):
    from rlinf.envs.sim.world_model.backend import bwm

    pipe = _RecordingBwmPipeline((4, 6))
    monkeypatch.setattr(bwm, "RobotwinActionBridge", _HalfActionBridge)
    monkeypatch.setattr(bwm.BwmBackend, "_build_pipeline", lambda self: pipe)
    cfg = OmegaConf.create(
        {
            "chunk": 50,
            "condition_frame_length": 9,
            "image_size": [4, 6],
            "num_inference_steps": 2,
            "action_dim": 14,
            "rest_action": [0.0] * 14,
            "action_bridge": {
                "urdf_path": "",
                "srdf_path": "",
                "stat_path": "",
                "robot_pose": [0.0] * 7,
            },
        }
    )
    backend = bwm.BwmBackend(cfg, torch.device("cpu"))
    reset_window = [torch.full((3, 1, 4, 6), -1.0)] * 9
    backend.open_session([0], [reset_window], torch.zeros(1, 9, 14), seeds=[7])

    frames = backend.generate([0], torch.ones(1, 50, 14))
    backend.generate([0], torch.full((1, 50, 14), 2.0))

    assert frames.shape == (1, 3, 50, 4, 6)
    first, second = pipe.calls
    assert (first["seed"], second["seed"]) == (7, 8)
    # Bridged history (rest pose), the chunk, then its last command held twice.
    assert first["action"][0, :9].eq(0.0).all()
    assert first["action"][0, 9:].eq(0.5).all()
    # The reset frame stays in slot 0; the last 8 generated frames and poses roll in.
    assert second["input_video"][0, :, 0].eq(-1.0).all()
    assert second["input_video"][0, :, 1:].eq(1.0).all()
    assert second["action"][0, 0].eq(0.0).all()
    assert second["action"][0, 1:9].eq(0.5).all()
    assert second["action"][0, 9:].eq(1.0).all()
