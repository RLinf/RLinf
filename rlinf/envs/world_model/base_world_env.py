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

"""Common utilities for world model based environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Union

import torch

from rlinf.scheduler import WorkerInfo


def _is_musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def _is_cuda_backend_available() -> bool:
    return torch.cuda.is_available() and hasattr(torch._C, "_cuda_setDevice")


class BaseWorldEnv(ABC):
    """Base class that provides shared utilities for world model environments.

    Subclasses are expected to implement dataset creation as well as any
    environment-specific logic such as model loading, stepping, and rendering.
    """

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: WorkerInfo,
        record_metrics: bool = True,
    ):
        self.cfg = cfg
        if _is_musa_available():
            self.device = torch.device("musa")
        elif _is_cuda_backend_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.num_envs = num_envs
        self.worker_info = worker_info
        self.record_metrics = record_metrics

        self.auto_reset = getattr(cfg, "auto_reset", True)
        self.ignore_terminations = getattr(cfg, "ignore_terminations", False)
        self.use_rel_reward = getattr(cfg, "use_rel_reward", False)

        self._is_start = True
        self.elapsed_steps = 0

        self.video_cfg = cfg.video_cfg

        self.prev_step_reward = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        # Whether to use KIR Trick
        self.enable_kir = cfg.get("enable_kir", True)

        self.dataset = self._build_dataset(cfg)

        if self.record_metrics:
            self._init_metrics()

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        if not hasattr(self, "_elapsed_steps"):
            self._elapsed_steps = 0
        return self._elapsed_steps

    @elapsed_steps.setter
    def elapsed_steps(self, value):
        self._elapsed_steps = value

    @abstractmethod
    def _build_dataset(self, cfg):
        """Return the dataset wrapper used for resets."""

    @abstractmethod
    def chunk_step(self, actions):
        """Advance the environment by one chunk and return (obs, reward, done, info)."""

    @abstractmethod
    def reset(self):
        """Reset the environment and return initial observations."""

    @abstractmethod
    def step(self, actions):
        """Perform a single action step and return (obs, reward, done, info)."""

    def _get_runtime_device_str(self) -> str:
        if self.device.type in {"cuda", "musa"}:
            device_index = 0 if self.device.index is None else self.device.index
            return f"{self.device.type}:{device_index}"
        return self.device.type

    @staticmethod
    def _clear_accelerator_cache() -> None:
        if _is_musa_available():
            torch.musa.empty_cache()
        elif _is_cuda_backend_available():
            torch.cuda.empty_cache()

    def _init_metrics(self):
        """Initialize episode metrics tensors."""
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx: Optional[Union[int, torch.Tensor]] = None):
        """Reset metrics either globally or for targeted environments."""
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
        self.elapsed_steps = 0

    def _record_metrics(self, step_reward, terminations, infos):
        """Store episode metrics inside the info dict."""
        if not self.record_metrics:
            return infos

        # Update success_once based on terminations
        if isinstance(terminations, torch.Tensor):
            self.success_once = self.success_once | terminations
        else:
            terminations_tensor = torch.tensor(
                terminations, device=self.device, dtype=torch.bool
            )
            self.success_once = self.success_once | terminations_tensor

        episode_info = {}
        self.returns += step_reward
        episode_info["return"] = self.returns.clone()
        infos["episode"] = episode_info
        return infos

    def update_reset_state_ids(self):
        """Optional hook to manage reset states for subclasses."""
        return
