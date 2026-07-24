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
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from rlinf.envs.gen_reward.datasets import EnvRecord, ImageConditionedDataset
from rlinf.envs.gen_reward.utils import media_to_uint8_nhwc


class LeRobotImageConditionedDataset(ImageConditionedDataset):
    """LeRobot -> text-image-to-video adapter."""

    default_image_keys = ("observation.images.front",)
    default_task_key = "task"
    default_action_key = "action"
    default_prompt_prefix = ""

    def __init__(
        self,
        dataset: Any,
        sample_mode: str,
        future_seconds: float,
        num_frames: int,
        prompt_prefix: str = "",
        action_key: str = "action",
    ):
        self.dataset = dataset
        self.sample_mode = sample_mode
        self.future_seconds = future_seconds
        self.num_frames = num_frames
        self.prompt_prefix = prompt_prefix
        self.action_key = action_key
        self.episode_ranges = self._episode_ranges()

    @classmethod
    def from_config(cls, cfg: Any) -> "LeRobotImageConditionedDataset":
        sample_mode = getattr(cfg, "sample_mode", "episode")
        future_seconds = float(cfg.future_seconds)
        if sample_mode not in ("episode", "frame"):
            raise ValueError(f"Unknown LeRobot sample_mode: {sample_mode}")
        if future_seconds < 0 and sample_mode != "episode":
            raise ValueError("future_seconds=-1 requires sample_mode='episode'.")

        dataset = LeRobotDataset(
            str(cfg.repo_id),
            root=cfg.root,
            episodes=getattr(cfg, "episodes", None),
            video_backend=getattr(cfg, "video_backend", "pyav"),
        )
        return cls(
            dataset=dataset,
            sample_mode=sample_mode,
            future_seconds=future_seconds,
            num_frames=int(cfg.num_frames),
            prompt_prefix=getattr(cfg, "prompt_prefix", cls.default_prompt_prefix),
            action_key=getattr(cfg, "action_key", cls.default_action_key),
        )

    def __len__(self) -> int:
        if self.sample_mode == "episode":
            return len(self.episode_ranges)
        return len(self.dataset)

    def __getitem__(self, index: int) -> EnvRecord:
        if self.sample_mode == "episode":
            start, end = self.episode_ranges[int(index)]
        else:
            start = int(index)
            end = self._episode_end_for_frame(start)

        if self.future_seconds < 0:
            frame_indices = np.linspace(start, end, self.num_frames)
        else:
            offsets = np.linspace(
                0.0, self.future_seconds * self.dataset.fps, self.num_frames
            )
            frame_indices = np.minimum(start + offsets, end)
        frame_indices = np.rint(frame_indices).astype(int)

        samples = [self.dataset[int(idx)] for idx in frame_indices]
        videos = []
        for key in self.default_image_keys:
            frames = [media_to_uint8_nhwc(sample[key])[0] for sample in samples]
            videos.append(np.stack(frames, axis=0))

        video = videos[0] if len(videos) == 1 else self.compose_videos(videos)
        sample0 = samples[0]
        task = sample0[self.default_task_key]
        record = {
            "task_description": self.format_task_description(str(task), sample0),
            "main_image": video[0],
            "future_video": video[1:] if video.shape[0] > 1 else None,
            "action": np.stack([sample[self.action_key] for sample in samples], axis=0),
        }
        record["arm_tag"] = sample0["arm_tag"]
        return record

    def format_task_description(self, task: str, sample: dict[str, Any]) -> str:
        return self.prompt_prefix + task

    def _episode_ranges(self) -> list[tuple[int, int]]:
        ranges = []
        current_episode = None
        start = 0
        episode_indices = self.dataset.hf_dataset["episode_index"]
        for local_index, episode_index in enumerate(episode_indices):
            episode_index = int(episode_index)
            if current_episode is None:
                current_episode = episode_index
                start = local_index
            elif episode_index != current_episode:
                ranges.append((start, local_index - 1))
                current_episode = episode_index
                start = local_index
        if current_episode is not None:
            ranges.append((start, len(self.dataset) - 1))
        return ranges

    def _episode_end_for_frame(self, index: int) -> int:
        for start, end in self.episode_ranges:
            if start <= index <= end:
                return end
        raise IndexError(f"Frame index {index} is outside the dataset episodes.")

    def compose_videos(self, videos: list[np.ndarray]) -> np.ndarray:
        num_frames = min(video.shape[0] for video in videos)
        return np.stack(
            [
                self.compose_views([video[idx] for video in videos])
                for idx in range(num_frames)
            ],
            axis=0,
        )

    def compose_views(self, views: list[np.ndarray]) -> np.ndarray:
        return views[0]


DATASET_CLS = LeRobotImageConditionedDataset


__all__ = ["DATASET_CLS", "LeRobotImageConditionedDataset"]
