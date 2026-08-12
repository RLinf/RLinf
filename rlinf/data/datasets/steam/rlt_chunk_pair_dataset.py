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

"""Chunk-aligned temporal pair view over RLT ManiSkill LeRobot data."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .binning import _signed_stride_to_bin
from .pair_dataset import _LeRobotSource, _resolve_alias, _to_uint8_hwc

_STATE_KEY_ALIASES = (
    "{key}",
    "observation/{key}",
    "observation.{key}",
)


class RLTChunkPairDataset(Dataset):
    """Yield positive/reversed STEAM pairs only at action-chunk boundaries."""

    def __init__(
        self,
        dataset_path: str,
        *,
        chunk_size: int = 10,
        k: int = 4,
        num_bins: int = 8,
        dataset_type: str = "sft",
        only_success: bool = True,
        main_image_key: str = "image",
        wrist_image_key: str = "wrist_image",
        state_key: str = "state",
        default_prompt: str = "insert the peg in the hole",
    ) -> None:
        self.chunk_size = int(chunk_size)
        self.k = int(k)
        self.num_bins = int(num_bins)
        if self.chunk_size < 1 or self.k < 1:
            raise ValueError("chunk_size and k must both be >= 1")
        if self.num_bins < 2 or self.num_bins % 2 != 0:
            raise ValueError("num_bins must be an even integer >= 2")
        if self.num_bins > 2 and (2 * self.k) % self.num_bins != 0:
            raise ValueError(
                "RLT chunk pair binning requires 2*k to be divisible by num_bins; "
                f"got k={self.k}, num_bins={self.num_bins}."
            )

        self.main_image_key = str(main_image_key)
        self.wrist_image_key = str(wrist_image_key)
        self.state_key = str(state_key)
        self.default_prompt = str(default_prompt)
        self.source_name = str(dataset_path)
        self._source = _LeRobotSource(
            dataset_path,
            only_success=bool(only_success),
            dataset_type=str(dataset_type),
        )
        self._rng: np.random.Generator | None = None

        self._anchors: list[tuple[int, int, int]] = []
        for episode in range(self._source.num_episodes()):
            if only_success and not self._source.episode_is_success(episode):
                continue
            episode_length = self._source.episode_length(episode)
            for frame_t in range(0, episode_length - 1, self.chunk_size):
                max_gap_chunks = (episode_length - 1 - frame_t) // self.chunk_size
                max_gap_chunks = min(self.k, max_gap_chunks)
                if self.num_bins == 2 and max_gap_chunks < self.k:
                    continue
                if max_gap_chunks >= 1:
                    self._anchors.append((episode, frame_t, max_gap_chunks))
        if not self._anchors:
            raise ValueError(
                "No complete chunk-aligned temporal pairs found in "
                f"{dataset_path!r} for chunk_size={self.chunk_size}."
            )

    def __len__(self) -> int:
        return 2 * len(self._anchors)

    def set_epoch(self, epoch: int) -> None:
        self._rng = np.random.default_rng(int(epoch))

    def _rng_for_worker(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    def _build_observation(
        self,
        *,
        raw_sample: dict[str, Any],
        episode: int,
        frame: int,
        prompt: str,
    ) -> dict[str, Any]:
        main_image = self._source.get_view_from_sample(raw_sample, self.main_image_key)
        wrist_image = self._source.get_view_from_sample(
            raw_sample, self.wrist_image_key
        )
        if main_image is None or wrist_image is None:
            raise KeyError(
                "RLT STEAM pairs require both main and wrist images; "
                f"episode={episode}, frame={frame}."
            )
        state = _resolve_alias(raw_sample, self.state_key, _STATE_KEY_ALIASES)
        if isinstance(state, torch.Tensor):
            state = state.detach().cpu().numpy()
        return {
            "main_images": _to_uint8_hwc(main_image),
            "wrist_images": _to_uint8_hwc(wrist_image),
            "states": np.asarray(state, dtype=np.float32),
            "task_descriptions": prompt,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        anchor_idx, is_positive = divmod(index, 2)
        episode, frame_t, max_gap_chunks = self._anchors[anchor_idx]
        gap_chunks = (
            self.k
            if self.num_bins == 2
            else int(self._rng_for_worker().integers(1, max_gap_chunks + 1))
        )
        frame_tk = frame_t + gap_chunks * self.chunk_size
        signed_gap = gap_chunks if is_positive == 0 else -gap_chunks
        first_frame, second_frame = (
            (frame_t, frame_tk) if signed_gap > 0 else (frame_tk, frame_t)
        )
        raw_first = self._source.get_raw_sample(episode, first_frame)
        raw_second = self._source.get_raw_sample(episode, second_frame)
        prompt = self.default_prompt or self._source.get_prompt_from_sample(
            raw_first, episode, first_frame
        )
        if self.num_bins == 2:
            label = 1 if signed_gap > 0 else 0
        else:
            label = _signed_stride_to_bin(signed_gap, self.k, self.num_bins)
        return {
            "obs_t": self._build_observation(
                raw_sample=raw_first,
                episode=episode,
                frame=first_frame,
                prompt=prompt,
            ),
            "obs_tk": self._build_observation(
                raw_sample=raw_second,
                episode=episode,
                frame=second_frame,
                prompt=prompt,
            ),
            "label": int(label),
            "episode": episode,
            "frame_idx_t": first_frame,
            "frame_idx_tk": second_frame,
        }


class RLTChunkPairCollator:
    """Collate raw RLT observation pairs for frozen Stage 1 extraction."""

    @staticmethod
    def _collate_observations(samples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "main_images": torch.from_numpy(
                np.stack([sample["main_images"] for sample in samples])
            ),
            "wrist_images": torch.from_numpy(
                np.stack([sample["wrist_images"] for sample in samples])
            ),
            "extra_view_images": None,
            "states": torch.from_numpy(
                np.stack([sample["states"] for sample in samples])
            ).to(dtype=torch.float32),
            "task_descriptions": [sample["task_descriptions"] for sample in samples],
        }

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "obs_t": self._collate_observations([sample["obs_t"] for sample in batch]),
            "obs_tk": self._collate_observations(
                [sample["obs_tk"] for sample in batch]
            ),
            "labels": torch.as_tensor(
                [sample["label"] for sample in batch], dtype=torch.long
            ),
        }


__all__ = ["RLTChunkPairCollator", "RLTChunkPairDataset"]
