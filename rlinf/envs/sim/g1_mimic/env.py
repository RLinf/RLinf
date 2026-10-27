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

"""SuperMimic simulator backends, sim context sampling, and episode-end codes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import torch

from rlinf.envs.sim.g1_mimic.constants import (
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    XI_DIM,
)

if TYPE_CHECKING:
    from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env


def sample_xi(num: int, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
    """Draw the 33-d sim context: foot friction, joint offsets, torso CoM offset."""
    xi = torch.zeros(num, XI_DIM, device=device, dtype=dtype)
    xi[:, 0] = torch.empty(num, device=device, dtype=dtype).uniform_(0.3, 1.2)
    xi[:, 1:30] = torch.empty(num, 29, device=device, dtype=dtype).uniform_(-0.01, 0.01)
    xi[:, 30] = torch.empty(num, device=device, dtype=dtype).uniform_(-0.025, 0.025)
    xi[:, 31:33] = torch.empty(num, 2, device=device, dtype=dtype).uniform_(-0.05, 0.05)
    return xi


def try_make_mjlab_env(**kwargs: Any) -> MjlabG1Env:
    """Load the mjlab backend when the optional extra is installed."""
    try:
        from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env
    except ImportError as exc:
        raise ImportError(
            "env.backend=mjlab requires the optional mjlab extra. "
            "Install from requirements/embodied/envs/supermimic.txt "
            "(commit 8bb2bc9)."
        ) from exc
    return MjlabG1Env(**kwargs)


def compose_end_kind(
    failure: torch.Tensor,
    timeout: torch.Tensor,
    clip_end: torch.Tensor,
) -> torch.Tensor:
    """Merge one task's failure with the episode limits.

    Failure outranks timeout, and timeout outranks clip end. ``failure`` is
    the active task's own condition: motion tracking, or a velocity fall.
    Timeout and clip end remain when that condition is false, so a velocity
    episode cannot turn an ended episode back into CONT.
    """
    end = torch.full(
        failure.shape, EPISODE_CONT, dtype=torch.long, device=failure.device
    )
    end = torch.where(clip_end, torch.full_like(end, EPISODE_CLIP_END), end)
    end = torch.where(timeout, torch.full_like(end, EPISODE_TIMEOUT), end)
    end = torch.where(failure, torch.full_like(end, EPISODE_FAILURE), end)
    return end


_BACKENDS: dict[str, Callable] = {"mjlab": try_make_mjlab_env}


def register_env_backend(name: str, factory: Callable) -> None:
    """Add a simulator backend in this process. Training configs use ``mjlab``."""
    _BACKENDS[name] = factory


def env_backends() -> tuple[str, ...]:
    """Names of the registered env backends."""
    return tuple(_BACKENDS)


def make_env(backend: str, **kwargs: Any) -> Any:
    """Build the env registered under ``backend`` with ``kwargs``.

    Args:
        backend: A name from ``env_backends()``.
        **kwargs: Passed to the backend factory.

    Returns:
        The env object the factory builds.

    Raises:
        ValueError: If no backend has that name.
    """
    if backend not in _BACKENDS:
        raise ValueError(
            f"unknown SuperMimic env backend {backend!r}; available: {sorted(_BACKENDS)}"
        )
    return _BACKENDS[backend](**kwargs)
