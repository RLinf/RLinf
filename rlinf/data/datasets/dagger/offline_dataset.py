# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static LeRobot datasets used as offline rehearsal data for DAgger."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from rlinf.data.lerobot_paths import resolve_lerobot_dataset_root
from rlinf.utils.logging import get_logger

logger = get_logger()


def _to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return value.item()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


class OfflineLeRobotDataset(Dataset):
    """Read a finalized LeRobot dataset lazily for DAgger rehearsal.

    Unlike :class:`RollingLeRobotDataset`, this dataset keeps the finalized
    GELLO/SFT dataset disk-backed. Images are decoded only for sampled frames,
    so adding a large offline dataset does not copy it into the online replay
    buffer or eagerly decode every image.

    Args:
        dataset_path: Full local LeRobot dataset directory containing
            ``meta/info.json`` and ``data/``.
        chunk_size: Number of future actions returned for each sample.
        target_fps: Temporal spacing of the action chunk. When set, offline
            actions use the nearest integer source-frame stride to this rate.
            This keeps the action horizon as close as possible to the online
            environment while satisfying LeRobot timestamp constraints.
        episodes: Optional episode indices to include.
        default_task: Fallback prompt if neither the sample nor dataset
            metadata contains a task description.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        chunk_size: int,
        target_fps: int | None = None,
        episodes: Sequence[int] | None = None,
        default_task: str | None = None,
    ) -> None:
        self.dataset_path = str(dataset_path)
        self.root = resolve_lerobot_dataset_root(self.dataset_path)
        if not (self.root / "meta" / "info.json").is_file():
            raise FileNotFoundError(
                "Offline DAgger dataset is not a finalized local LeRobot "
                f"dataset: {self.root}"
            )
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if target_fps is not None and int(target_fps) <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")

        # Keep this import lazy so online-only DAgger does not require LeRobot
        # dataset initialization on the actor process.
        from lerobot.common.datasets import lerobot_dataset

        self.metadata = lerobot_dataset.LeRobotDatasetMetadata(
            self.dataset_path,
            root=self.root,
        )
        features = getattr(self.metadata, "features", {})
        if "actions" in features:
            self._source_action_key = "actions"
        elif "action" in features:
            self._source_action_key = "action"
        else:
            raise ValueError(
                "Offline DAgger dataset must contain an 'actions' or 'action' "
                f"feature, got {list(features)} from {self.root}"
            )

        source_fps = int(getattr(self.metadata, "fps", 0))
        if source_fps <= 0:
            raise ValueError(
                f"Offline DAgger dataset has invalid fps={source_fps}: {self.root}"
            )
        self.target_fps = int(target_fps) if target_fps is not None else source_fps
        self.frame_stride = max(1, int(round(source_fps / self.target_fps)))
        self.effective_fps = source_fps / self.frame_stride
        delta_timestamps = {
            self._source_action_key: [
                step * self.frame_stride / source_fps for step in range(int(chunk_size))
            ]
        }
        self._base = lerobot_dataset.LeRobotDataset(
            self.dataset_path,
            root=self.root,
            episodes=list(episodes) if episodes is not None else None,
            delta_timestamps=delta_timestamps,
        )
        self._tasks = self._normalize_tasks(getattr(self.metadata, "tasks", {}))
        self.default_task = default_task

        logger.info(
            "[OfflineLeRobotDataset] root=%s, samples=%d, source_fps=%d, "
            "target_fps=%d, effective_fps=%.3f, frame_stride=%d, "
            "chunk_size=%d, episodes=%s",
            self.root,
            len(self._base),
            source_fps,
            self.target_fps,
            self.effective_fps,
            self.frame_stride,
            int(chunk_size),
            list(episodes) if episodes is not None else "all",
        )

    @staticmethod
    def _normalize_tasks(tasks: Any) -> dict[int, str]:
        if isinstance(tasks, Mapping):
            return {int(index): str(task) for index, task in tasks.items()}
        if isinstance(tasks, Sequence) and not isinstance(tasks, (str, bytes)):
            return {index: str(task) for index, task in enumerate(tasks)}
        return {}

    def __len__(self) -> int:
        return len(self._base)

    def _normalize_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(sample)
        if "actions" not in item and self._source_action_key in item:
            item["actions"] = item[self._source_action_key]

        if "task" not in item:
            task_index = _to_python_scalar(item.get("task_index"))
            if task_index is not None:
                item["task"] = self._tasks.get(int(task_index), "")
            else:
                item["task"] = ""
        if not item["task"] and self.default_task is not None:
            item["task"] = self.default_task

        missing = [key for key in ("state", "actions") if key not in item]
        if missing:
            raise KeyError(
                f"Offline DAgger sample from {self.root} is missing {missing}; "
                f"available keys are {sorted(item)}"
            )
        return item

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._normalize_sample(self._base[int(index)])

    def __getitems__(self, indices: Sequence[int]) -> list[dict[str, Any]]:
        return [self[int(index)] for index in indices]

    def get_stats(self) -> dict[str, Any]:
        return {
            "samples": len(self),
            "source_fps": int(getattr(self.metadata, "fps", 0)),
            "target_fps": self.target_fps,
            "effective_fps": self.effective_fps,
            "frame_stride": self.frame_stride,
        }
