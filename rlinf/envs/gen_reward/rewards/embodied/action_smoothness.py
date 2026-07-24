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

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rlinf.envs.gen_reward.rewards import FRAME_LEVEL, VIDEO_LEVEL
from rlinf.envs.gen_reward.rewards.embodied.action_prediction import (
    ActionPredictionRewardBackend,
)
from rlinf.envs.gen_reward.utils import cfg_get


class ActionSmoothnessRewardBackend(ActionPredictionRewardBackend):
    """EVA-style IDM action smoothness reward for generated videos."""

    supported_reward_levels = (FRAME_LEVEL, VIDEO_LEVEL)
    reward_type = FRAME_LEVEL

    @classmethod
    def _from_config(cls, cfg: Any) -> "ActionSmoothnessRewardBackend":
        backend = super()._from_config(cfg)
        backend.dt = 1.0 / float(cfg_get(cfg, "fps", 30.0))
        backend.joint_name = cfg_get(cfg, "return_joint_name", None)
        return backend

    def _score_video(
        self,
        output_video: np.ndarray,
        record: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        actions = self._predict_actions(output_video)
        if self.reward_type == VIDEO_LEVEL:
            reward = self._video_action_smoothness(actions).repeat(
                output_video.shape[0]
            )
        elif self.reward_type == FRAME_LEVEL:
            reward = self._frame_action_smoothness(actions)

        scores = {"avg": reward, "action_smoothness": reward}
        if self.joint_name == "idm":
            action_tensor = actions.detach().cpu()
        elif self.joint_name == "gt":
            action_tensor = torch.as_tensor(record["action"]).float()
        else:
            action_tensor = None

        if action_tensor is not None:
            for joint_idx in range(action_tensor.shape[-1]):
                scores[f"joint_{joint_idx}"] = action_tensor[..., joint_idx]
        return scores

    def _frame_action_smoothness(self, actions: torch.Tensor) -> torch.Tensor:
        num_frames = actions.shape[0]
        if num_frames < 4:
            return torch.zeros(
                (num_frames,), device=actions.device, dtype=torch.float32
            )

        vlim_pen, acc_pen, jerk_pen, alim_pen = self._smoothness_penalties(actions)
        vlim_frame = torch.empty(num_frames, device=actions.device, dtype=actions.dtype)
        acc_frame = torch.empty_like(vlim_frame)
        jerk_frame = torch.empty_like(vlim_frame)
        alim_frame = torch.empty_like(vlim_frame)

        vlim_frame[1:] = vlim_pen
        vlim_frame[0] = vlim_pen[0]
        acc_frame[2:] = acc_pen
        acc_frame[:2] = acc_pen[0]
        alim_frame[2:] = alim_pen
        alim_frame[:2] = alim_pen[0]
        jerk_frame[3:] = jerk_pen
        jerk_frame[:3] = jerk_pen[0]

        penalty = 0.5 * jerk_frame + 0.25 * acc_frame + 2.0 * vlim_frame + alim_frame
        return 10.0 * torch.pow(1.0 + penalty / 2000.0, -0.5)

    def _video_action_smoothness(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[0] < 4:
            return torch.zeros((), device=actions.device, dtype=torch.float32)

        vlim_pen, acc_pen, jerk_pen, alim_pen = self._smoothness_penalties(actions)
        penalty = (
            0.5 * jerk_pen.mean()
            + 0.25 * acc_pen.mean()
            + 2.0 * vlim_pen.mean()
            + alim_pen.mean()
        )
        return 10.0 * torch.pow(1.0 + penalty / 2000.0, -0.5)

    def _smoothness_penalties(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actions = actions.float()
        dt = self.dt
        dims = actions.shape[-1]
        device = actions.device
        dtype = actions.dtype

        v_arm = (
            torch.tensor(
                [3.1416, 3.4034, 3.1416, 3.9270, 3.9270, 3.9270],
                device=device,
                dtype=dtype,
            )
            * 0.85
        )
        v_max = torch.zeros(dims, device=device, dtype=dtype)
        v_max[0:6] = v_arm
        v_max[7:13] = v_arm
        v_max[6] = 2.0
        v_max[13] = 2.0

        a_max = torch.full((dims,), 5.0, device=device, dtype=dtype)
        a_max[6] = 10.0
        a_max[13] = 10.0

        v = (actions[1:] - actions[:-1]) / dt
        a = (v[1:] - v[:-1]) / dt
        j = (a[1:] - a[:-1]) / dt

        weights = torch.ones(dims, device=device, dtype=dtype)
        weights[6] = 0.2
        weights[13] = 0.2
        weights = weights / (weights.mean() + 1e-8)

        acc_pen = (self._huber(a, a_max * 0.5) * weights).mean(dim=-1)
        jerk_pen = (self._huber(j, (a_max / dt) * 0.5) * weights).mean(dim=-1)
        vlim_pen = ((torch.relu(v.abs() - v_max) ** 2) * weights).mean(dim=-1)
        alim_pen = ((torch.relu(a.abs() - a_max) ** 2) * weights).mean(dim=-1)
        return vlim_pen, acc_pen, jerk_pen, alim_pen

    @staticmethod
    def _huber(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        abs_x = x.abs()
        return torch.where(
            abs_x <= delta, 0.5 * x.pow(2), delta * (abs_x - 0.5 * delta)
        )


REWARD_CLS = ActionSmoothnessRewardBackend


__all__ = ["ActionSmoothnessRewardBackend", "REWARD_CLS"]
