# Copyright 2025 The RLinf Authors.
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

"""Gym wrappers that compute rewards via a visual classifier."""

from __future__ import annotations

from typing import Any, Callable, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
import torch
from gymnasium.core import ActType, ObsType


class ClassifierRewardWrapper(gym.Wrapper):
    """Replace the environment reward with a visual-classifier prediction.

    At each step the wrapper feeds the camera frames through a classifier
    function, applies sigmoid to obtain a success probability, and returns
    a binary reward (``1.0`` if the probability exceeds the threshold,
    ``0.0`` otherwise).  The raw probability is stored in
    ``info["classifier_reward"]``.  An episode is terminated when the
    classifier predicts success.

    Args:
        env: The base environment.
        classifier_func: A callable ``obs_dict → float`` that returns
            the classifier logit given the current observation.
        reward_threshold: Sigmoid probability threshold for declaring success.
    """

    def __init__(
        self,
        env: gym.Env,
        classifier_func: Callable[[dict], float],
        reward_threshold: float = 0.5,
        override_termination: bool = True,
    ) -> None:
        super().__init__(env)
        self._classifier_func = classifier_func
        self._reward_threshold = reward_threshold
        self._override_termination = override_termination
        self._step_count = 0

    def step(
        self, action: ActType,
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._step_count += 1
        obs, _rew, done, truncated, info = self.env.step(action)
        raw_logit = float(self._classifier_func(obs))
        prob = _sigmoid(raw_logit)
        is_success = prob >= self._reward_threshold
        if self._override_termination:
            done = is_success
        else:
            done = done or is_success
        info["succeed"] = is_success
        info["classifier_reward"] = prob
        return obs, float(is_success), done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["succeed"] = False
        return obs, info


class MultiStageClassifierRewardWrapper(gym.Wrapper):
    """Multi-stage visual reward using a sequence of classifiers.

    Each classifier guards a successive stage.  Once stage *i* fires
    (``sigmoid(logit) >= stage_threshold``), the agent receives a
    partial reward and moves on.  The episode terminates when all stages
    are completed.

    Args:
        env: The base environment.
        classifier_funcs: Ordered list of classifier callables.
        stage_rewards: Reward granted when each stage is completed.
            Defaults to evenly-spaced values ending at 1.
        stage_threshold: Sigmoid probability threshold per stage.
    """

    def __init__(
        self,
        env: gym.Env,
        classifier_funcs: list[Callable[[dict], float]],
        stage_rewards: Optional[list[float]] = None,
        stage_threshold: float = 0.75,
        override_termination: bool = True,
    ) -> None:
        super().__init__(env)
        self._classifier_funcs = classifier_funcs
        n = len(classifier_funcs)
        if stage_rewards is not None:
            assert len(stage_rewards) == n
            self._stage_rewards = list(stage_rewards)
        else:
            self._stage_rewards = [(i + 1) / n for i in range(n)]
        self._stage_threshold = stage_threshold
        self._override_termination = override_termination
        self._completed: list[bool] = [False] * n

    def step(self, action):
        obs, _rew, done, truncated, info = self.env.step(action)

        reward = 0.0
        for i, func in enumerate(self._classifier_funcs):
            if self._completed[i]:
                continue
            prob = _sigmoid(func(obs))
            if prob >= self._stage_threshold:
                self._completed[i] = True
                reward += self._stage_rewards[i]

        all_done = all(self._completed)
        if self._override_termination:
            done = all_done
        else:
            done = done or all_done
        info["succeed"] = all_done
        info["classifier_stages_completed"] = sum(self._completed)
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._completed = [False] * len(self._classifier_funcs)
        info["succeed"] = False
        return obs, info


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def make_classifier_reward_func(
    classifier_model: torch.nn.Module,
    image_keys: list[str],
    device: str | torch.device = "cpu",
    obs_key: str = "frames",
) -> Callable[[dict], float]:
    """Create a callable that extracts camera frames and returns a reward.

    The returned function takes the full observation dict from the
    environment and feeds the relevant camera images through the
    classifier.

    Args:
        classifier_model: A trained :class:`RewardClassifier` (eval mode).
        image_keys: Camera keys inside ``obs[obs_key]``.
        device: Inference device.
        obs_key: Top-level key under which camera frames live.

    Returns:
        A function ``obs_dict → float`` returning the classifier logit.
    """

    @torch.no_grad()
    def _reward_func(obs: dict) -> float:
        frames = obs[obs_key]
        batch = {}
        for key in image_keys:
            img = frames[key]
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.ndim == 3:
                img = img.unsqueeze(0)
            batch[key] = img.to(device)
        logit = classifier_model(batch)
        return float(logit.item())

    return _reward_func


def make_remote_classifier_reward_func(
    server_name: str = "ClassifierRewardServer",
    obs_key: str = "frames",
) -> Callable[[dict], float]:
    """Create a reward function backed by a remote :class:`ClassifierRewardServer`.

    The server must already be running as a named Ray actor.  Each call
    ships the camera frames to the server and blocks until the logit is
    returned.

    Args:
        server_name: Name of the :class:`ClassifierRewardServer` actor.
        obs_key: Top-level key in the observation dict containing frames.

    Returns:
        A function ``obs_dict → float`` returning the classifier logit.
    """
    import ray

    server = ray.get_actor(server_name)

    def _remote_reward_func(obs: dict) -> float:
        # Serialize as raw bytes to avoid numpy pickle version mismatch
        # between nodes (numpy 1.x vs 2.x).
        frames_raw: dict[str, tuple[bytes, tuple, str]] = {}
        for key, val in sorted(obs[obs_key].items()):
            arr = val
            if isinstance(arr, torch.Tensor):
                arr = arr.cpu().numpy()
            arr = np.ascontiguousarray(arr)
            frames_raw[key] = (arr.tobytes(), arr.shape, str(arr.dtype))
        logit: float = ray.get(server.predict.remote(frames_raw))
        return logit

    return _remote_reward_func
