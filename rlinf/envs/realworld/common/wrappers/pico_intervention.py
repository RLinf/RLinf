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

"""PICO intervention wrapper for single-arm Franka environments."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.pico.pico_expert import PicoExpert


class PicoIntervention(gym.ActionWrapper):
    """Override policy actions with PICO controller input while deadman is held."""

    def __init__(
        self,
        env,
        gripper_enabled: bool = True,
        **pico_config: Any,
    ):
        super().__init__(env)
        self.gripper_enabled = gripper_enabled
        self.expert = PicoExpert(**pico_config)
        self.left = False
        self.right = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.left = False
        self.right = False
        return obs, info

    def _match_env_action_space(
        self,
        action: np.ndarray,
        fallback_action: np.ndarray,
    ) -> np.ndarray:
        """Pad/truncate PICO actions to the wrapped env's action dimension."""
        target_shape = self.env.action_space.shape
        target_dim = int(np.prod(target_shape))
        action_flat = np.asarray(action, dtype=np.float32).reshape(-1)

        if action_flat.size == target_dim:
            return action_flat.reshape(target_shape)

        matched = np.zeros((target_dim,), dtype=np.float32)
        copy_dim = min(action_flat.size, target_dim)
        matched[:copy_dim] = action_flat[:copy_dim]

        fallback_flat = np.asarray(fallback_action, dtype=np.float32).reshape(-1)
        if action_flat.size < target_dim and fallback_flat.size >= target_dim:
            matched[action_flat.size : target_dim] = fallback_flat[
                action_flat.size : target_dim
            ]

        return matched.reshape(target_shape)

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool, dict[str, Any]]:
        tcp_pose = self.get_wrapper_attr("get_tcp_pose")()
        action_scale = self.get_wrapper_attr("get_action_scale")()

        expert_action, replaced, pico_info = self.expert.get_action(
            tcp_pose,
            action_scale,
            gripper_enabled=self.gripper_enabled,
        )
        self.left = bool(
            pico_info.get("pico_active", False) and pico_info.get("pico_hand") == "left"
        )
        self.right = bool(
            pico_info.get("pico_active", False)
            and pico_info.get("pico_hand") == "right"
        )

        if replaced:
            expert_action = self._match_env_action_space(expert_action, action)
            return expert_action, True, pico_info
        return action, False, pico_info

    def step(self, action):
        new_action, replaced, pico_info = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        info.update(pico_info)
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        self.expert.stop()
        return super().close()
