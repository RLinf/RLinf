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

from rlinf.envs.gen_reward.rewards import (
    FRAME_LEVEL,
    VIDEO_LEVEL,
    VideoRewardBackendBase,
)
from rlinf.envs.gen_reward.rewards.generation.ocr import OCRScorer
from rlinf.envs.gen_reward.utils import extract_quoted_text


class VideoOCRRewardBackend(VideoRewardBackendBase):
    """Video OCR reward with frame-level or video-level output mode."""

    supported_reward_levels = (FRAME_LEVEL, VIDEO_LEVEL)
    reward_type = FRAME_LEVEL

    def __init__(self, scorer: OCRScorer):
        self.scorer = scorer

    @classmethod
    def _from_config(cls, cfg: Any) -> "VideoOCRRewardBackend":
        return cls(scorer=OCRScorer.from_config(cfg))

    def _score_video(
        self,
        media: np.ndarray,
        record: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        target = extract_quoted_text(record["task_description"])
        frame_scores = torch.tensor(
            [self.scorer.score_image(frame, target) for frame in media],
            dtype=torch.float32,
        )
        if self.reward_type == VIDEO_LEVEL:
            reward = frame_scores.mean().repeat(media.shape[0])
        elif self.reward_type == FRAME_LEVEL:
            reward = frame_scores
        return {"avg": reward, "video_ocr": reward}


REWARD_CLS = VideoOCRRewardBackend


__all__ = ["REWARD_CLS", "VideoOCRRewardBackend"]
