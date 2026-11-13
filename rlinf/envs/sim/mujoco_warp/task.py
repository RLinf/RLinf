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

"""Task wrappers over a MuJoCo Warp world."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch

from rlinf.envs.sim.mujoco_warp.world import MjWarpWorld

__all__ = ["TaskWrapper"]


class TaskWrapper(ABC):
    """Observation, reward, termination, and reset state for one task.

    ``step`` writes the action, runs the task's physics substeps, and returns
    the observation, reward, and termination. ``reset`` writes the initial
    state of the selected worlds after those worlds have been cleared.
    """

    @abstractmethod
    def obs_dim(self, world: MjWarpWorld) -> int:
        """Return the observation width for ``world``."""

    @abstractmethod
    def frame_skip(self) -> int:
        """Return the number of physics steps in one control step."""

    @abstractmethod
    def observe(self, world: MjWarpWorld) -> torch.Tensor:
        """Return the current observation for every world."""

    @abstractmethod
    def reset(
        self,
        world: MjWarpWorld,
        world_ids: torch.Tensor,
        reset_state_ids: np.ndarray,
    ) -> torch.Tensor:
        """Reset ``world_ids`` and return their observations.

        Args:
            world: World that owns the physics state.
            world_ids: 1-D indices of the worlds to reset.
            reset_state_ids: One deterministic reset id per selected world.

        Returns:
            Observations for ``world_ids`` only.
        """

    @abstractmethod
    def step(
        self, world: MjWarpWorld, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Advance every world by one control step.

        Args:
            world: World that owns the physics state.
            actions: Control batch with shape ``(num_envs, nu)``.

        Returns:
            Observations, rewards, and terminations for every world.
        """
