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


class ActionSimilarityRewardBackend(ActionPredictionRewardBackend):
    """IDM-based action similarity reward for image-conditioned videos."""

    supported_reward_levels = (FRAME_LEVEL, VIDEO_LEVEL)
    reward_type = FRAME_LEVEL

    def _score_video(
        self,
        output_video: np.ndarray,
        record: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        pred_actions = self._normalize_actions(self._predict_actions(output_video))
        target_actions = torch.as_tensor(
            record["action"],
            device=self.device,
            dtype=torch.float32,
        )
        target_actions = self._normalize_actions(target_actions)
        mse = (pred_actions - target_actions).pow(2).mean(dim=-1)
        frame_similarity = torch.exp(-mse / self.temperature).clamp(0.0, 1.0)
        if self.reward_type == VIDEO_LEVEL:
            reward = frame_similarity.mean().repeat(output_video.shape[0])
        elif self.reward_type == FRAME_LEVEL:
            reward = frame_similarity
        return {"avg": reward, "action_similarity": reward}


REWARD_CLS = ActionSimilarityRewardBackend


__all__ = ["ActionSimilarityRewardBackend", "REWARD_CLS"]
