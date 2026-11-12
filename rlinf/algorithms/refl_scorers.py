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

"""Differentiable scorers used by ReFL actors."""

from collections.abc import Callable
from typing import Any, Protocol

import torch


class ReFLScorer(Protocol):
    """Score samples while preserving their gradient path.

    Implementations freeze their model parameters and return rewards of shape
    [B], with larger values preferred. Sample layout and any required context
    in ``forward_inputs`` are owned by the implementation.
    """

    def __call__(
        self, samples: torch.Tensor, forward_inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Return per-sample rewards without detaching ``samples``."""
        ...


ReFLScorerBuilder = Callable[[Any, torch.device | str], ReFLScorer]
_SCORER_REGISTRY: dict[str, ReFLScorerBuilder] = {}


def register_refl_scorer(name: str, builder: ReFLScorerBuilder) -> None:
    """Register a scorer builder in each process that constructs ReFL actors."""
    if not name:
        raise ValueError("ReFL scorer name must be a non-empty string.")
    if not callable(builder):
        raise TypeError("ReFL scorer builder must be callable.")
    if name in _SCORER_REGISTRY:
        raise ValueError(f"ReFL scorer {name!r} is already registered.")
    _SCORER_REGISTRY[name] = builder


def build_refl_scorer(cfg: Any, device: torch.device | str) -> ReFLScorer:
    """Build the configured scorer, defaulting to ImageReward."""
    name = cfg.get("type", "image_reward")
    if name not in _SCORER_REGISTRY:
        raise ValueError(
            f"Unknown ReFL scorer {name!r}; expected one of {sorted(_SCORER_REGISTRY)}."
        )
    return _SCORER_REGISTRY[name](cfg, device)


def _build_image_reward(cfg: Any, device: torch.device | str) -> ReFLScorer:
    from rlinf.envs.sim.diffusion.rewards.visual.image_reward import ImageRewardScorer

    return ImageRewardScorer.from_config(cfg, device)


register_refl_scorer("image_reward", _build_image_reward)
