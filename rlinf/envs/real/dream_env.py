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

"""Real-world control and observation capture for Dream evaluation."""

import time
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.real.env import (
    ChunkStepResult,
    EnvInfos,
    Observation,
    RealWorldEnv,
    StepResult,
)
from rlinf.envs.wrappers.dream import DreamObservationBuffer
from rlinf.scheduler import WorkerInfo


class DreamRealEnv(RealWorldEnv):
    """Capture control observations and pad terminal episodes until reset."""

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: WorkerInfo | None,
    ) -> None:
        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)
        self._observations = DreamObservationBuffer()
        self._origin: float | None = None
        self._terminal: StepResult | None = None
        self._commanded: torch.Tensor | None = None

    def reset(self, **kwargs: Any) -> tuple[Observation, EnvInfos]:
        """Reset and collect nine camera ticks without advancing control steps."""
        observation, info = super().reset(**kwargs)
        self._observations.reset()
        self._origin = None
        self._terminal = None
        self._commanded = None
        self._capture(observation)
        for _ in range(8):
            time.sleep(0.2)
            raw = self.env.envs[0].unwrapped.observe()
            observation = self._wrap_obs(
                {
                    group: {name: value[None] for name, value in values.items()}
                    for group, values in raw.items()
                }
            )
            self._capture(observation)
        return observation, {**info, "dream_observations": self._pack()}

    def step(self, actions: Any = None, auto_reset: bool = False) -> StepResult:
        """Run a control step or return the cached terminal transition."""
        if self._terminal is not None:
            obs, reward, terminated, truncated, info = self._terminal
            return obs, torch.zeros_like(reward), terminated, truncated, dict(info)
        try:
            result = super().step(actions, auto_reset=False)
            observation, _, terminated, truncated, info = result
            self._capture(observation)
            self._commanded = torch.as_tensor(info["accepted_action"]).clone()
            if bool((terminated | truncated).any()):
                self._hold()
                self._terminal = (*result[:4], dict(info))
            return result
        except Exception:
            try:
                self._hold()
            except Exception as error:
                from rlinf.utils.logging import get_logger

                get_logger().warning("Dream hold failed (%s).", type(error).__name__)
            raise

    def chunk_step(self, chunk_actions: np.ndarray) -> ChunkStepResult:
        """Attach encoded control observations to the final chunk info."""
        self._observations.clear()
        result = super().chunk_step(chunk_actions)
        result[-1][-1]["dream_observations"] = self._pack()
        return result

    def _hold(self) -> None:
        self.env.envs[0].unwrapped.emergency_hold()

    def _capture(self, observation: Observation) -> None:
        captured = time.monotonic()
        if self._origin is None:
            self._origin = captured
        self._observations.append(observation, captured - self._origin)

    def _pack(self) -> dict[str, Any]:
        return self._observations.pack(
            self.task_descriptions,
            self._commanded,
            finished=self._terminal is not None,
        )
