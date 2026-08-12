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

"""Chunk-aligned STEAM pairs read directly from ManiSkill RLT data."""

from typing import Any, Optional, Sequence

import numpy as np
from torch.utils.data import Dataset

from .binning import _scaled_signed_stride_to_bin, _signed_stride_to_bin
from .pair_dataset import _LeRobotSource


class RLTChunkPairDataset(Dataset):
    """Expose ManiSkill RLT trajectories as native STEAM frame pairs.

    The on-disk dataset remains the RLT LeRobot dataset with ``image``,
    ``wrist_image``, ``state``, and ``actions`` fields. STEAM consumes only
    the two image views and task instruction. Temporal anchors and strides are
    measured in RLT action chunks while returned frame indices remain raw
    trajectory indices.
    """

    def __init__(
        self,
        dataset_path: str,
        *,
        camera_keys: Sequence[str] = ("image", "wrist_image"),
        chunk_size: int = 10,
        k: int = 4,
        num_bins: int = 8,
        dataset_type: str = "sft",
        only_success: bool = True,
        default_prompt: str = "insert the peg in the hole",
        length_scale_enabled: bool = True,
        length_scale_percentile: float = 100.0,
        length_scale_reference: Optional[float] = None,
    ) -> None:
        self.camera_keys = tuple(str(key) for key in camera_keys)
        if not self.camera_keys:
            raise ValueError("camera_keys must be non-empty")

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

        self.default_prompt = str(default_prompt)
        self.source_name = str(dataset_path)
        self.length_scale_enabled = bool(length_scale_enabled)
        self.length_scale_percentile = float(length_scale_percentile)
        if not 0.0 < self.length_scale_percentile <= 100.0:
            raise ValueError("length_scale_percentile must be in (0, 100]")
        self._length_scale_reference = (
            None if length_scale_reference is None else float(length_scale_reference)
        )

        self._source = _LeRobotSource(
            dataset_path,
            only_success=bool(only_success),
            dataset_type=str(dataset_type),
        )
        self._rng: np.random.Generator | None = None
        self._eligible: list[int] = []
        self._episode_chunk_lengths: dict[int, int] = {}
        self._anchors: list[tuple[int, int, int]] = []

        for episode in range(self._source.num_episodes()):
            if only_success and not self._source.episode_is_success(episode):
                continue
            episode_length = self._source.episode_length(episode)
            episode_chunk_length = (episode_length - 1) // self.chunk_size + 1
            if episode_chunk_length < 2:
                continue
            self._eligible.append(episode)
            self._episode_chunk_lengths[episode] = episode_chunk_length
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
        if self.length_scale_enabled and self._length_scale_reference is None:
            self._length_scale_reference = self._compute_length_scale_reference()

    def __len__(self) -> int:
        return 2 * len(self._anchors)

    def set_epoch(self, epoch: int) -> None:
        self._rng = np.random.default_rng(int(epoch))

    def _rng_for_worker(self) -> np.random.Generator:
        if self._rng is None:
            self._rng = np.random.default_rng()
        return self._rng

    @property
    def length_scale_reference(self) -> Optional[float]:
        """Return the reference trajectory length in action chunks."""
        return self._length_scale_reference

    def eligible_episode_lengths(self) -> list[int]:
        """Return eligible trajectory lengths measured in action chunks."""
        return [self._episode_chunk_lengths[ep] for ep in self._eligible]

    def _compute_length_scale_reference(self) -> float:
        lengths = np.asarray(self.eligible_episode_lengths(), dtype=np.float64)
        return max(1.0, float(np.percentile(lengths, self.length_scale_percentile)))

    def set_length_scale_reference(self, reference: float) -> None:
        """Set the shared STEAM trajectory-length normalization reference."""
        if not self.length_scale_enabled:
            return
        if reference <= 0:
            raise ValueError(f"length_scale_reference must be > 0, got {reference}")
        self._length_scale_reference = float(reference)

    def _load_views(
        self,
        raw_sample: dict[str, Any],
        *,
        episode: int,
        frame: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
        views: dict[str, np.ndarray] = {}
        masks: dict[str, bool] = {}
        for camera_key in self.camera_keys:
            view = self._source.get_view_from_sample(raw_sample, camera_key)
            masks[camera_key] = view is not None
            if view is not None:
                views[camera_key] = view
        if not views:
            raise KeyError(
                "RLT STEAM pair has no configured image views; "
                f"episode={episode}, frame={frame}, camera_keys={self.camera_keys}."
            )
        return views, masks

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)

        anchor_idx = index // 2
        is_positive = index % 2 == 0
        episode, frame_t, max_gap_chunks = self._anchors[anchor_idx]
        gap_chunks = (
            self.k
            if self.num_bins == 2
            else int(self._rng_for_worker().integers(1, max_gap_chunks + 1))
        )
        frame_tk = frame_t + gap_chunks * self.chunk_size
        signed_gap = gap_chunks if is_positive else -gap_chunks
        first_frame, second_frame = (
            (frame_t, frame_tk) if is_positive else (frame_tk, frame_t)
        )

        raw_first, raw_second = self._source.get_raw_pair(
            episode,
            first_frame,
            second_frame,
            camera_keys=self.camera_keys,
        )
        prompt = self.default_prompt
        if not prompt:
            prompt = self._source.get_prompt_from_sample(
                raw_first,
                episode,
                first_frame,
            )

        if self.num_bins == 2:
            label = 1 if is_positive else 0
        elif self.length_scale_enabled and self._length_scale_reference is not None:
            episode_length = self._episode_chunk_lengths[episode]
            scaled_gap = signed_gap * self._length_scale_reference / episode_length
            label = _scaled_signed_stride_to_bin(
                scaled_gap,
                self.k,
                self.num_bins,
            )
        else:
            label = _signed_stride_to_bin(signed_gap, self.k, self.num_bins)

        views_t, masks_t = self._load_views(
            raw_first,
            episode=episode,
            frame=first_frame,
        )
        views_tk, masks_tk = self._load_views(
            raw_second,
            episode=episode,
            frame=second_frame,
        )
        return {
            "image_t": views_t,
            "image_tk": views_tk,
            "image_mask_t": masks_t,
            "image_mask_tk": masks_tk,
            "prompt": prompt,
            "label": int(label),
            "episode": int(episode),
            "frame_idx_t": int(first_frame),
            "frame_idx_tk": int(second_frame),
            "source_name": self.source_name,
        }


__all__ = ["RLTChunkPairDataset"]
