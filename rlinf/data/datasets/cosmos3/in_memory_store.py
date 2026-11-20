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

"""Store observed camera windows alongside online DAgger action chunks."""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from rlinf.data.datasets.dagger.in_memory_store import InMemoryArrowStore


class Cosmos3InMemoryArrowStore(InMemoryArrowStore):
    """Add same-episode video windows and optional CHW uint8 camera storage."""

    def __init__(
        self,
        chunk_size: int = 1,
        action_sequence_keys: list[str] | None = None,
        fps: int = 10,
        image_transforms: Callable | None = None,
        vision_horizon: int = 0,
        keep_uint8_images: bool = False,
    ) -> None:
        super().__init__(chunk_size, action_sequence_keys, fps, image_transforms)
        self._vision_horizon = vision_horizon
        self._keep_uint8_images = keep_uint8_images
        self._episode_images: deque[dict[str, torch.Tensor]] = deque()

    def _infer_hf_features(self, frame: dict) -> tuple[Any, set[str]]:
        features, image_keys = super()._infer_hf_features(frame)
        if self._keep_uint8_images:
            from datasets import Features

            # The cached pixels live outside Arrow; all other columns use LeRobot.
            features = Features(
                {
                    key: feature
                    for key, feature in features.items()
                    if key not in image_keys
                }
            )
        return features, image_keys

    def _prepare_episode_dataset(
        self,
        ep_frames: list[dict],
        base: int | None = None,
        episode_index: int | None = None,
    ) -> dict[str, Any]:
        prepared = super()._prepare_episode_dataset(ep_frames, base, episode_index)
        if not prepared:
            return prepared

        images = {}
        if self._keep_uint8_images:
            for key in prepared["image_keys"]:
                pixels = np.stack([frame[key] for frame in ep_frames])
                images[key] = torch.from_numpy(pixels).permute(0, 3, 1, 2).contiguous()
        prepared["images"] = images
        return prepared

    def _commit_prepared_episode(self, prepared_episode: dict[str, Any]) -> int:
        if not prepared_episode:
            return 0

        images = prepared_episode["images"]
        num_frames = super()._commit_prepared_episode(prepared_episode)
        self._episode_images.append(images)
        return num_frames

    def __getitem__(self, local_idx: int) -> dict[str, Any]:
        item = super().__getitem__(local_idx)
        if not self._keep_uint8_images and not self._vision_horizon:
            return item

        local_idx = max(0, min(local_idx, self._total_frames - 1))
        ep_idx = bisect.bisect_right(self._ep_to, local_idx)
        local_frame = local_idx - self._ep_from[ep_idx]
        images = self._episode_images[ep_idx]
        for key, frames in images.items():
            item[key] = frames[local_frame].clone()
        if images and self._image_transforms is not None:
            for key in self._image_keys:
                item[key] = self._image_transforms(item[key])

        if self._vision_horizon:
            valid = local_idx + self._vision_horizon < self._ep_to[ep_idx]
            self._add_vision_window(
                item, self._episode_datasets[ep_idx], images, local_frame, valid
            )
        return item

    def _add_vision_window(
        self,
        item: dict[str, Any],
        ep_ds: Any,
        images: dict[str, torch.Tensor],
        local_frame: int,
        valid: bool,
    ) -> None:
        image_to_window = {"image": "vision_images"}
        if "wrist_image" in self._image_keys:
            image_to_window["wrist_image"] = "vision_wrist_images"

        item["vision_valid"] = valid
        if not valid:
            for image_key, window_key in image_to_window.items():
                item[window_key] = (
                    item[image_key]
                    .unsqueeze(0)
                    .expand(self._vision_horizon + 1, -1, -1, -1)
                )
            return

        end = local_frame + self._vision_horizon + 1
        if images:
            for image_key, window_key in image_to_window.items():
                item[window_key] = images[image_key][local_frame:end].clone()
            return

        future_frames = ep_ds.select_columns(list(image_to_window))[
            local_frame + 1 : end
        ]
        for image_key, window_key in image_to_window.items():
            frames = future_frames[image_key]
            if self._image_transforms is not None:
                frames = [self._image_transforms(frame) for frame in frames]
            item[window_key] = torch.stack([item[image_key], *frames])
