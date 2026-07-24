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

from rlinf.envs.gen_reward.utils import (
    cfg_get,
    cfg_require,
    media_to_uint8_nhwc,
    normalize_type,
)

FRAME_LEVEL = "frame_level"
VIDEO_LEVEL = "video_level"
REWARD_LEVELS = (FRAME_LEVEL, VIDEO_LEVEL)

RewardOutputs = torch.Tensor | np.ndarray | list[Any]
RewardRecords = list[dict[str, Any]]
RewardScores = dict[str, torch.Tensor]


class RewardBackend:
    """Base interface for generated-output reward backends.

    Args:
        outputs: Generated images/videos from the rollout side.
        records: Env records returned by `dataset.build_grouped_env_batch`.

    Returns:
        Score dict containing the configured `reward.key`, usually `avg`.
        Image rewards use shape [B]. Video rewards use shape [B, T];
        `VideoRewardBackendBase` expands video-level scalar rewards to [T]
        before the env maps frames to latent/action reward chunks.
    """

    supported_reward_levels = REWARD_LEVELS
    reward_type = VIDEO_LEVEL

    @classmethod
    def from_config(cls, cfg: Any) -> "RewardBackend":
        supported = tuple(cls.supported_reward_levels)
        reward_type = cfg_get(cfg, "reward_type", cls.reward_type)
        if reward_type not in supported:
            raise ValueError(
                f"{cls.__name__} supports reward_type {supported}, got {reward_type}."
            )

        backend = cls._from_config(cfg)
        backend.reward_type = reward_type
        return backend

    @classmethod
    def _from_config(cls, cfg: Any) -> "RewardBackend":
        raise NotImplementedError

    def score(
        self,
        outputs: RewardOutputs,
        records: RewardRecords,
    ) -> RewardScores:
        raise NotImplementedError


class ImageRewardBackendBase(RewardBackend):
    """Base helper for rewards that score one image with one scalar."""

    supported_reward_levels = (VIDEO_LEVEL,)
    reward_type = VIDEO_LEVEL

    def to_image_batch(self, outputs: RewardOutputs) -> np.ndarray:
        images = media_to_uint8_nhwc(outputs)
        if images.ndim != 4:
            raise ValueError(
                f"Expected image batch [B,H,W,C], got shape {images.shape}."
            )
        return images


class VideoRewardBackendBase(RewardBackend):
    """Base helper for rewards that score generated videos over time."""

    supported_reward_levels = (FRAME_LEVEL, VIDEO_LEVEL)
    reward_type = FRAME_LEVEL

    def to_video_batch(self, outputs: RewardOutputs) -> np.ndarray:
        videos = media_to_uint8_nhwc(outputs)
        if videos.ndim == 4:
            videos = videos[:, None]
        if videos.ndim != 5:
            raise ValueError(
                f"Expected video batch [B,T,H,W,C], got shape {videos.shape}."
            )
        return videos

    def score(
        self,
        outputs: RewardOutputs,
        records: RewardRecords,
    ) -> RewardScores:
        videos = self.to_video_batch(outputs)
        score_rows = {}
        for video, record in zip(videos, records, strict=True):
            row_scores = self._score_video(video, record)
            for name, reward in row_scores.items():
                if torch.is_tensor(reward):
                    reward_tensor = reward.detach().cpu().float().reshape(-1)
                else:
                    reward_tensor = torch.as_tensor(
                        reward, dtype=torch.float32
                    ).reshape(-1)

                if reward_tensor.numel() != video.shape[0]:
                    raise ValueError(
                        f"{name} reward must have {video.shape[0]} values, "
                        f"got {reward_tensor.numel()}."
                    )
                score_rows.setdefault(name, []).append(reward_tensor)
        return {name: torch.stack(rows).float() for name, rows in score_rows.items()}

    def _score_video(
        self,
        video: np.ndarray,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError


class MultiRewardBackend(RewardBackend):
    def __init__(self, reward_backends: list[tuple[str, float, RewardBackend]]):
        self.reward_backends = reward_backends

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        build_single_reward_backend,
    ) -> "MultiRewardBackend":
        reward_backends = []
        for reward_cfg in cfg_require(cfg, "rewards"):
            reward_model = normalize_type(cfg_require(reward_cfg, "model"))
            name = str(cfg_get(reward_cfg, "name", reward_model.split(".")[-1]))
            weight = float(cfg_get(reward_cfg, "weight", 1.0))
            reward_backends.append(
                (name, weight, build_single_reward_backend(reward_cfg))
            )
        return cls(reward_backends)

    def score(
        self,
        outputs: RewardOutputs,
        records: RewardRecords,
    ) -> RewardScores:
        scores = {}
        weighted_rewards = []
        total_weight = 0.0
        for name, weight, backend in self.reward_backends:
            backend_scores = backend.score(outputs, records)
            weighted_rewards.append(backend_scores["avg"].float() * weight)
            total_weight += weight
            for key, value in backend_scores.items():
                scores[f"{name}.{key}"] = value
        scores["avg"] = sum(weighted_rewards) / total_weight
        return scores


__all__ = [
    "FRAME_LEVEL",
    "VIDEO_LEVEL",
    "ImageRewardBackendBase",
    "MultiRewardBackend",
    "RewardBackend",
    "RewardOutputs",
    "RewardRecords",
    "RewardScores",
    "VideoRewardBackendBase",
]
