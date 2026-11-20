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

"""Rolling LeRobot windows for online Cosmos3 training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rlinf.data.datasets.dagger.dataset import RollingLeRobotDataset
from rlinf.data.datasets.dagger.lerobot_io import _hf_column_to_numpy_bool_1d
from rlinf.models.embodiment.cosmos3.datasets.in_memory_store import (
    Cosmos3InMemoryArrowStore,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


class Cosmos3RollingLeRobotDataset(RollingLeRobotDataset):
    """Extend action chunks with same-episode future camera frames.

    Args:
        vision_horizon: Number of future frames after the current observation.
        keep_uint8_images: Cache decoded uint8 camera frames in each shard.
        require_full_window: Limit sampling to complete action and vision windows.
    """

    def __init__(
        self,
        *args: Any,
        vision_horizon: int = 0,
        keep_uint8_images: bool = False,
        require_full_window: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._vision_horizon = vision_horizon
        self._keep_uint8_images = keep_uint8_images
        self.require_full_window = bool(require_full_window)
        if self.require_full_window and self._valid_physical_indices is None:
            self._valid_physical_indices = []
            self._valid_physical_set = set()

    def _new_in_memory_store(self) -> Cosmos3InMemoryArrowStore:
        return Cosmos3InMemoryArrowStore(
            chunk_size=self._shard_cache_chunk_size,
            action_sequence_keys=self._shard_cache_action_keys,
            fps=self._shard_cache_fps,
            image_transforms=self.image_transforms,
            vision_horizon=self._vision_horizon,
            keep_uint8_images=self._keep_uint8_images,
        )

    def append_episode_to_memory(self, path: str | Path, ep_frames: list[dict]) -> None:
        if not ep_frames:
            return
        path = Path(path)
        with self._rolling_access_lock:
            if path in self._in_memory_shards:
                ds_idx = self._sub_datasets.index(path)
                if ds_idx != len(self._sub_datasets) - 1:
                    raise ValueError(
                        "Can only append to the latest in-memory LeRobot shard."
                    )
                store = self._in_memory_shards[path]
            else:
                store = self._new_in_memory_store()
            old_len = len(store)
            episode_index = store.num_episodes

        # Build the expensive HuggingFace episode outside the sampling lock.
        prepared_episode = store._prepare_episode_dataset(
            ep_frames,
            base=old_len,
            episode_index=episode_index,
        )
        if not prepared_episode:
            return

        with self._rolling_access_lock:
            if path in self._in_memory_shards:
                ds_idx = self._sub_datasets.index(path)
                if ds_idx != len(self._sub_datasets) - 1:
                    raise ValueError(
                        "Can only append to the latest in-memory LeRobot shard."
                    )
                store = self._in_memory_shards[path]
                physical_base = self._cumulative_lengths[ds_idx]
            else:
                self._in_memory_shards[path] = store
                self._sub_datasets.append(path)
                physical_base = self._cumulative_lengths[-1]
                self._cumulative_lengths.append(physical_base)

            old_len = int(prepared_episode["base"])
            added_frames = store._commit_prepared_episode(prepared_episode)
            if added_frames <= 0:
                return

            self._cumulative_lengths[-1] += added_frames
            self._total_episodes += 1
            if self._valid_physical_indices is not None:
                ep_ds = prepared_episode["dataset"]
                for offset in self._valid_episode_offsets(ep_ds, added_frames):
                    gidx = physical_base + old_len + int(offset)
                    self._valid_physical_indices.append(gidx)
                    self._valid_physical_set.add(gidx)

            self._update_window_sampling_bounds()
            self._evict_stale_shards()
            logger.debug(
                "[RollingLeRobotDataset] episode appended: %s "
                "(+%d frames, physical_frames=%d, logical_samples=%d)",
                path.name,
                added_frames,
                self._num_physical_frames(),
                len(self),
            )

    def _evict_stale_shards(self) -> int:
        """Evict shards before the sampling window while keeping the latest writable."""
        if not self._in_memory_shards or not self._window_enabled():
            return 0
        n_evicted = 0
        for ds_idx, path in enumerate(self._sub_datasets):
            # The latest shard can receive episodes even if none is usable yet.
            if ds_idx == len(self._sub_datasets) - 1:
                continue
            shard_end = self._cumulative_lengths[ds_idx + 1]
            if (
                shard_end <= self._window_physical_start
                and self._in_memory_shards.pop(path, None) is not None
            ):
                n_evicted += 1
        if n_evicted:
            logger.debug(
                "[RollingLeRobotDataset] evicted %d shard(s) from shard cache "
                "(window_physical_start=%d)",
                n_evicted,
                self._window_physical_start,
            )
        return n_evicted

    def _valid_episode_offsets(self, episode: Any, num_frames: int) -> np.ndarray:
        """Return eligible starts without crossing an episode boundary."""
        num_starts = num_frames
        if self.require_full_window:
            window_length = max(int(self.chunk_size), self._vision_horizon + 1)
            num_starts = max(0, num_frames - window_length + 1)
        offsets = np.arange(num_starts, dtype=np.int64)
        if not self.require_all_intervene or not num_starts:
            return offsets
        if self.intervene_flag_key not in episode.column_names:
            logger.warning(
                "[RollingLeRobotDataset] require_all_intervene=True but column "
                "%r missing; keeping all %d eligible starts for this episode.",
                self.intervene_flag_key,
                num_starts,
            )
            return offsets
        flags = _hf_column_to_numpy_bool_1d(episode, self.intervene_flag_key)
        raw = offsets[:, None] + np.arange(max(1, int(self.chunk_size)))[None, :]
        in_episode = raw < num_frames
        step_ok = np.ones_like(in_episode, dtype=np.bool_)
        step_ok[in_episode] = flags[raw[in_episode]]
        return offsets[step_ok.all(axis=1)]

    def _append_valid_indices_for_store(
        self, store: Cosmos3InMemoryArrowStore, physical_base: int
    ) -> None:
        """Add eligible indices for one attached in-memory store."""
        if self._valid_physical_indices is None:
            return
        # Resume rebuilds the index under the sampling lock, starting at frame zero.
        if physical_base == 0:
            self._valid_physical_indices.clear()
            self._valid_physical_set.clear()

        for ep_ds, ep_from, ep_to in zip(
            store._episode_datasets,
            store._ep_from,
            store._ep_to,
            strict=True,
        ):
            for offset in self._valid_episode_offsets(ep_ds, int(ep_to) - int(ep_from)):
                gidx = int(physical_base) + int(ep_from) + int(offset)
                self._valid_physical_indices.append(gidx)
                self._valid_physical_set.add(gidx)

    def get_stats(self) -> dict[str, Any]:
        return {**super().get_stats(), "require_full_window": self.require_full_window}
