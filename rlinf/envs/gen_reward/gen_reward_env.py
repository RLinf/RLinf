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

import gym
import numpy as np
import torch

from . import build_reward_backend, build_reward_dataset
from .rewards import RewardBackend
from .utils import (
    cfg_get,
    cfg_require,
    make_future_video_comparison,
    media_to_uint8_nhwc,
    put_video_text,
)


class GenRewardEnv(gym.Env):
    """One-step generated-output reward environment.

    Reset returns dataset context. Step receives generated outputs and returns rewards.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
    ):
        # config
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        base_seed = int(cfg_get(cfg, "seed", 42))
        self.seed = base_seed + self.seed_offset
        self.group_size = int(cfg_get(cfg, "group_size", 1))
        self.num_group = max(1, int(np.ceil(self.num_envs / max(1, self.group_size))))
        self.is_eval = bool(cfg_get(cfg, "is_eval", False))
        self._generator = np.random.default_rng(seed=self.seed)
        self._cursor = 0
        # dataset and reward backend
        self.dataset = build_reward_dataset(cfg_require(cfg, "dataset"))
        reward_cfg = cfg_require(cfg, "reward")
        self.reward_key = str(cfg_get(reward_cfg, "key", "avg"))
        self.frame_interval = int(cfg_get(reward_cfg, "frame_interval", -1))
        self.is_multi_reward_backend = (
            str(cfg_get(reward_cfg, "type", "single")) == "multi"
        )
        self.reward_backend: RewardBackend = build_reward_backend(reward_cfg)
        self.image_frame_repeat = 9
        self.num_capture_samples = 2
        self._return_video: np.ndarray | None = None
        self._env_records: list[dict[str, Any]] = []  # current dataset records
        self._env_obs: dict[str, Any] | None = None  # observation of the dataset

    def update_reset_state_ids(self):
        return

    def _next_group_indices(self) -> np.ndarray:
        # TODO: comment for eval visual
        # if self.is_eval:
        #     start = self._cursor + self.seed_offset * self.num_group
        #     self._cursor += self.num_group * self.total_num_processes
        #     indices = np.arange(start, start + self.num_group)
        #     return indices % len(self.dataset)
        return self._generator.integers(
            0,
            len(self.dataset),
            size=(self.num_group,),
        )

    def reset(self, *args, **kwargs) -> tuple[dict[str, Any], dict[str, Any]]:
        self._return_video = None
        group_indices = self._next_group_indices()
        self._env_obs, self._env_records = self.dataset.build_grouped_env_batch(
            group_indices=group_indices,
            group_size=self.group_size,
            num_envs=self.num_envs,
        )
        return self._env_obs, {}

    def step(
        self, outputs: torch.Tensor | np.ndarray | list[Any]
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        scores = self.reward_backend.score(outputs, self._env_records)
        task_descriptions = self._env_obs.get("task_descriptions")
        self._return_video = self._prepare_capture_media(
            outputs,
            self._env_records,
            task_descriptions,
            scores,
        )
        # return info
        rewards = scores[self.reward_key].float()
        if rewards.ndim > 1:
            rewards = torch.as_tensor(
                self._frame_rewards_to_env_steps(
                    rewards.detach().cpu().numpy(),
                ),
                dtype=rewards.dtype,
                device=rewards.device,
            )
        truncations = torch.zeros_like(rewards, dtype=torch.bool)
        terminations = torch.ones_like(rewards, dtype=torch.bool)
        if rewards.ndim > 1:
            terminations[..., :-1] = False
        episode = {
            "return": rewards.detach().float(),
            "episode_len": torch.ones(self.num_envs, dtype=torch.float32),
        }
        for key, value in scores.items():
            episode[key] = value.detach().float()
        episode[self.reward_key] = rewards.detach().float()
        final_obs = self._env_obs
        next_obs = self._env_obs
        infos = {
            "episode": episode,
            "final_info": {"episode": episode},
            "final_observation": final_obs,
        }
        return next_obs, rewards, terminations, truncations, infos

    def capture_image(self) -> np.ndarray | None:
        return self._return_video

    def _prepare_capture_media(
        self,
        media: torch.Tensor | np.ndarray | list[Any],
        records: list[dict[str, Any]],
        task_descriptions: list[str] | None = None,
        scores: dict[str, torch.Tensor] | None = None,
    ) -> np.ndarray | None:
        # mode selection
        media = media_to_uint8_nhwc(media)
        has_future_video = bool(records and records[0].get("future_video") is not None)
        if media.ndim == 4:
            mode = "image"
        elif has_future_video:
            mode = "embodied_video"
        else:
            mode = "video"
        # video pre-process
        if mode == "image":
            media = np.repeat(
                media[: self.num_capture_samples, None],
                self.image_frame_repeat,
                axis=1,
            )
        elif mode == "embodied_video":
            media = media[: self.num_capture_samples]
            media = make_future_video_comparison(
                media,
                records[: media.shape[0]],
            )
        else:
            media = media[: self.num_capture_samples]
        # plot the score curve
        score_curves = {}
        if scores is not None:
            for key, value in scores.items():
                if self.is_multi_reward_backend and not (
                    key == "avg" or key.endswith(".avg")
                ):
                    continue
                if torch.is_tensor(value):
                    value = value.detach().cpu().float().numpy()
                else:
                    value = np.asarray(value, dtype=np.float32)
                if value.ndim == 1:
                    value = np.repeat(value[:, None], media.shape[1], axis=1)
                if value.ndim == 2:
                    score_curves[key] = value[: media.shape[0]]

        return put_video_text(media, task_descriptions, score_curves)

    def chunk_step(
        self, outputs: torch.Tensor | np.ndarray | list[Any]
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        obs, rewards, terminations, truncations, infos = self.step(outputs)
        if rewards.ndim == 1:
            rewards = rewards.unsqueeze(1)
            terminations = terminations.unsqueeze(1)
            truncations = truncations.unsqueeze(1)
        return ([obs], rewards, terminations, truncations, [infos])

    def _frame_rewards_to_env_steps(self, frame_rewards: np.ndarray) -> np.ndarray:
        frame_rewards = np.asarray(frame_rewards, dtype=np.float32)
        if self.frame_interval <= 0:
            return frame_rewards

        chunks = [frame_rewards[..., :1]]
        for start in range(1, frame_rewards.shape[-1], self.frame_interval):
            chunk = frame_rewards[..., start : start + self.frame_interval]
            chunks.append(chunk.mean(axis=-1, keepdims=True))
        return np.concatenate(chunks, axis=-1).astype(np.float32)
