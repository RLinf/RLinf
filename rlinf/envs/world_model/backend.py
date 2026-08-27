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

"""The interface between a world-model environment and the thing that generates frames.

The env owns episode semantics — session lifetime, rewards, auto reset, metrics — and a backend only
advances frames. It is session-shaped so a backend that pools per-trajectory state knows when a
trajectory begins and ends. Implementations live next to this module, one file each.
"""

from __future__ import annotations

from typing import Any, Protocol, Sequence, runtime_checkable

import torch

__all__ = ["WorldModelBackend", "FrameQueue"]

FrameQueue = Sequence[Sequence[torch.Tensor]]


@runtime_checkable
class WorldModelBackend(Protocol):
    """Advances frames for a world-model environment."""

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        task_ids: Sequence[Any],
        seeds: Sequence[int],
    ) -> None:
        """Start a trajectory per env slot."""

    def generate(
        self,
        env_ids: Sequence[int],
        actions: torch.Tensor,
        condition: FrameQueue,
    ) -> torch.Tensor:
        """Advance one action chunk.

        Args:
            env_ids: Env slots the batch rows belong to, in row order.
            actions: ``[B, T, action_dim]``.
            condition: Per env slot, the condition frames as ``[C, 1, H, W]`` tensors.

        Returns:
            Generated frames as ``[B, C, T, H, W]`` in ``[-1, 1]``.
        """

    def close_session(self, env_ids: Sequence[int]) -> None:
        """End the trajectories of these env slots; slots without a session are ignored."""

    def offload(self) -> None:
        """Move weights off the execution device."""

    def onload(self) -> None:
        """Move weights back to the execution device."""
