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

"""RecordVideo frames come from each step's observation, not the live scene."""

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.wrappers.record_video import RecordVideo

CHUNK_SIZE = 3
RESET_VALUE = 200


def _video_cfg(tmp_path):
    return OmegaConf.create({"video_base_dir": str(tmp_path), "info_on_video": False})


def _obs(value: int) -> dict:
    return {"main_images": torch.full((1, 4, 4, 3), value, dtype=torch.uint8)}


class _ChunkEnv(gym.Env):
    """Simulator stand-in whose ``capture_image`` shows only the live scene."""

    num_envs = 1
    seed = 0

    def __init__(self, terminal: bool):
        self.terminal = terminal
        self.scene_value = 0

    def capture_image(self):
        return np.full((4, 4, 3), self.scene_value, dtype=np.uint8)

    def chunk_step(self, actions):
        obs_list = [_obs(10 * (i + 1)) for i in range(CHUNK_SIZE)]
        infos = [{} for _ in range(CHUNK_SIZE)]
        if self.terminal:
            # Auto reset: the last observation is the reset scene and the
            # terminal one moves to ``final_observation``.
            infos[-1]["final_observation"] = obs_list[-1]
            obs_list[-1] = _obs(RESET_VALUE)
            self.scene_value = RESET_VALUE
        else:
            self.scene_value = 10 * CHUNK_SIZE
        done = torch.zeros((1, CHUNK_SIZE), dtype=torch.bool)
        done[:, -1] = self.terminal
        return (
            obs_list,
            torch.zeros((1, CHUNK_SIZE)),
            done,
            torch.zeros((1, CHUNK_SIZE), dtype=torch.bool),
            infos,
        )


def _frame_values(env: RecordVideo) -> list[int]:
    return [int(frame.reshape(-1)[0]) for frame in env.render_images]


def test_chunk_frames_follow_each_step_observation(tmp_path):
    env = RecordVideo(_ChunkEnv(terminal=False), _video_cfg(tmp_path))
    env.chunk_step(None)
    assert _frame_values(env) == [10, 20, 30]


def test_terminal_frame_is_kept_before_the_reset_frame(tmp_path):
    env = RecordVideo(_ChunkEnv(terminal=True), _video_cfg(tmp_path))
    env.chunk_step(None)
    assert _frame_values(env) == [10, 20, 30, RESET_VALUE]


def test_capture_image_is_used_when_obs_has_no_image(tmp_path):
    inner = _ChunkEnv(terminal=False)
    inner.scene_value = 77
    env = RecordVideo(inner, _video_cfg(tmp_path))
    env.add_new_frames({"states": torch.zeros(1, 2)})
    assert _frame_values(env) == [77]
