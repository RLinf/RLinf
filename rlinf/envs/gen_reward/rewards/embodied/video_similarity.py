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

from rlinf.envs.gen_reward.rewards import (
    FRAME_LEVEL,
    VIDEO_LEVEL,
    VideoRewardBackendBase,
)
from rlinf.envs.gen_reward.utils import prepare_video_pair


class VideoSimilarityRewardBackend(VideoRewardBackendBase):
    """Reference-video similarity reward for video generation datasets."""

    supported_reward_levels = (FRAME_LEVEL, VIDEO_LEVEL)
    reward_type = FRAME_LEVEL

    @classmethod
    def _from_config(cls, cfg: Any) -> "VideoSimilarityRewardBackend":
        return cls()

    def _score_video(
        self,
        output_video: np.ndarray,
        record: dict[str, Any],
    ) -> dict[str, np.ndarray]:
        output_video, target_video = prepare_video_pair(output_video, record)
        mse = np.mean(
            (output_video.astype(np.float32) - target_video.astype(np.float32)) ** 2,
            axis=(1, 2, 3),
        )
        frame_similarity = np.clip(1.0 - mse / (255.0**2), 0.0, 1.0).astype(np.float32)
        if self.reward_type == VIDEO_LEVEL:
            reward = np.full(
                output_video.shape[0],
                float(frame_similarity.mean()),
                dtype=np.float32,
            )
        elif self.reward_type == FRAME_LEVEL:
            reward = frame_similarity
        return {"avg": reward, "video_similarity": reward}


REWARD_CLS = VideoSimilarityRewardBackend


__all__ = ["REWARD_CLS", "VideoSimilarityRewardBackend"]
