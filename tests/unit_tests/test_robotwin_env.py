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

"""RoboTwin observation and action contract at the simulator boundary."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.sim.robotwin.dream_env import DreamRobotwinEnv


class _Camera:
    def load_camera(self, scene):
        pass


class _Task:
    def __init__(self):
        self.tick = 0
        self.scene = SimpleNamespace(step=self.step, get_timestep=lambda: 0.125)
        self.robot = SimpleNamespace()
        for side, start, finger in (("left", 0, 0.5), ("right", 6, 1.0)):
            joints = list(range(7))
            entity = SimpleNamespace(
                get_active_joints=lambda joints=joints: joints,
                get_qpos=lambda start=start, finger=finger: [
                    *(np.arange(6) + start + self.tick),
                    finger,
                ],
            )
            setattr(self.robot, f"{side}_entity", entity)
            setattr(self.robot, f"{side}_arm_joints", joints[:6])
            setattr(self.robot, f"{side}_gripper", [(joints[-1], 0.5, 0.5)])
            setattr(self.robot, f"{side}_gripper_scale", (-1.0, 1.0))

    def step(self):
        self.tick += 1

    def get_info(self):
        return {"{a}": "Left"}

    def get_obs(self):
        return {"joint_action": {"vector": np.full(14, -1, dtype=np.float32)}}


class _VectorEnv:
    def __init__(self, **kwargs):
        self.envs = [SimpleNamespace(task=_Task())]

    def reset(self, **kwargs):
        self.envs[0].task = _Task()

    def get_obs(self):
        raw = self.envs[0].task.get_obs()
        rgb = np.zeros((8, 12, 3), dtype=np.uint8)
        return [
            {
                "full_image": rgb,
                "left_wrist_image": rgb,
                "right_wrist_image": rgb,
                "state": raw["joint_action"]["vector"].copy(),
                "instruction": "press the bell",
            }
        ]

    def step_with_full_obs(self, actions):
        self.actions = actions
        observations = []
        for _ in actions[0]:
            self.envs[0].task.scene.step()
            observations.append(self.get_obs())
        return observations, [0.0], [False], [False], [{"success": False}]


def test_dream_robotwin_measures_state_and_clips_last_chunk(monkeypatch):
    monkeypatch.setitem(sys.modules, "envs.camera", SimpleNamespace(Camera=_Camera))
    monkeypatch.setitem(
        sys.modules, "robotwin.envs.vector_env", SimpleNamespace(VectorEnv=_VectorEnv)
    )
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/env/robotwin_click_bell.yaml"
    )
    cfg.max_episode_steps = 3
    cfg.skip_intermediate_renders = False
    env = DreamRobotwinEnv(cfg, 1, 0, 1, None)
    _, info = env.reset()
    initial = info["dream_observations"]
    expected = torch.tensor([*range(6), 0.5, *range(6, 12), 1.0])
    torch.testing.assert_close(initial["states"][0, 0], expected)
    assert initial["active_arms"] == [("left",)]

    actions = torch.arange(56, dtype=torch.float32).reshape(1, 4, 14)
    env.chunk_step(actions[:, :2])
    observations, _, terminated, truncated, infos = env.chunk_step(actions[:, 2:])
    assert len(observations) == 1
    assert truncated.tolist() == [[True]] and not terminated.any()
    np.testing.assert_array_equal(env.venv.actions, actions[:, 2:3].numpy())

    packed = infos[-1]["dream_observations"]
    expected[:6] += 3
    expected[7:13] += 3
    torch.testing.assert_close(packed["states"][0, 0], expected)
    torch.testing.assert_close(packed["commanded"], actions[:, 2])
    assert packed["timestamps"].tolist() == [[0.375]]
    assert packed["episode_id"].tolist() == initial["episode_id"].tolist()
    assert packed["task_descriptions"] == ["press the bell"]
    assert packed["finished"].tolist() == [True]
    assert set(packed["frames"][0][0]) == {
        "video.overhead",
        "video.wrist_l",
        "video.wrist_r",
    }
