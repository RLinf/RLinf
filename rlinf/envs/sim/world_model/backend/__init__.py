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

The env owns episode semantics; a backend advances frames, loads the reward
model that scores them, and may supply per-frame task instructions. The
condition window lives behind the session, so the env hands it over once at
``open_session`` and afterwards sends only the action chunk. In-process
implementations live in this package, one module each (``wan``, ``opensora``).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, ContextManager, Optional, Protocol, Sequence

import torch

__all__ = ["WorldModelBackend", "FrameQueue", "autocast"]

FrameQueue = Sequence[Sequence[torch.Tensor]]


def autocast(device: torch.device, dtype: torch.dtype) -> ContextManager:
    """Autocast on an accelerator, a no-op on CPU."""
    if device.type == "cpu":
        return nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


class WorldModelBackend(Protocol):
    """Advances frames for a world-model environment."""

    # Whether the model conditions on KIR keyframes; ``enable_kir`` is rejected otherwise.
    supports_kir: bool
    chunk: int
    condition_frame_length: int
    image_size: tuple[int, int]

    @staticmethod
    def load_reward_model(cfg: Any) -> torch.nn.Module:
        """Return the reward model that scores generated frames."""

    def reward_instructions(self, env: Any) -> Optional[list[str]]:
        """Per-frame task instructions, or ``None`` if the reward model ignores the task."""

    def open_session(
        self,
        env_ids: Sequence[int],
        init_frames: FrameQueue,
        init_actions: torch.Tensor,
        seeds: Sequence[int],
    ) -> None:
        """Start a trajectory per env slot from its initial condition window.

        Args:
            env_ids: Env slots to open.
            init_frames: Per env slot, the initial condition frames as ``[C, 1, H, W]``
                tensors in ``[-1, 1]``. The first is the reference frame, kept for the
                whole trajectory.
            init_actions: ``[B, window, action_dim]``, the actions that led to
                ``init_frames``. A backend that conditions on the action chunk alone
                ignores them.
            seeds: Per env slot, the seed its noise is drawn from. A backend that draws
                noise from the global RNG ignores them.
        """

    def generate(
        self,
        env_ids: Sequence[int],
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Advance one action chunk from each session's own condition window.

        Args:
            env_ids: Env slots to advance; each needs an open session.
            actions: ``[B, chunk, action_dim]``, rows in ``env_ids`` order.

        Returns:
            The newly generated frames as ``[B, C, T, H, W]`` in ``[-1, 1]``. ``T``, the
            device and the dtype are the backend's own; the caller moves the result to
            where it keeps observations.
        """

    def close_session(self, env_ids: Sequence[int]) -> None:
        """End these env slots' trajectories; slots without a session are ignored."""

    def offload(self) -> None:
        """Move weights off the execution device."""

    def onload(self) -> None:
        """Move weights back to the execution device."""
