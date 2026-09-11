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

"""Interface between a world-model env and the backend that generates frames.

The env owns episode semantics; a backend only advances frames. The condition window
lives behind the session, so the env hands it over once at ``open_session`` and
afterwards sends only the action chunk. Implementations live next to this module, one
file each.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ContextManager, Protocol, Sequence, runtime_checkable

import torch

__all__ = ["WorldModelBackend", "FrameQueue", "autocast"]

FrameQueue = Sequence[Sequence[torch.Tensor]]


def autocast(device: torch.device, dtype: torch.dtype) -> ContextManager:
    """Autocast on an accelerator, a no-op on CPU."""
    if device.type == "cpu":
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


@runtime_checkable
class WorldModelBackend(Protocol):
    """Advances frames for a world-model environment."""

    chunk: int
    condition_frame_length: int
    image_size: tuple[int, int]

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        task_ids: Sequence[Any],
        seeds: Sequence[int],
    ) -> None:
        """Start a trajectory per env slot from its initial condition window.

        Args:
            env_ids: Env slots to open.
            init_frames: Per env slot, the initial condition frames as ``[C, 1, H, W]``
                tensors in ``[-1, 1]``. The first is the reference frame, kept for the
                whole trajectory.
            init_actions: ``[B, window, action_dim]``, the actions that led to
                ``init_frames``.
            task_ids: Per env slot, the task the trajectory runs.
            seeds: Per env slot, the seed its noise is drawn from.
        """

    def generate(
        self,
        env_ids: Sequence[int],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one action chunk from the session's own condition window."""

    def close_session(self, env_ids: Sequence[int]) -> None:
        """End these env slots' trajectories; slots without a session are ignored."""

    def offload(self) -> None:
        """Move weights off the execution device."""

    def onload(self) -> None:
        """Move weights back to the execution device."""
