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

from typing import Any, Protocol, runtime_checkable

import torch

from rlinf.models.embodiment.prefix.types import PrefixObs


@runtime_checkable
class PrefixFeatureModel(Protocol):
    """Frozen model that extracts prefix features and reference actions."""

    @property
    def z_dim(self) -> int: ...

    def extract_prefix_obs(self, env_obs: dict[str, Any]) -> PrefixObs:
        """Extract the compact observation consumed by the trainable policy."""
        ...


@runtime_checkable
class TrainablePrefixPolicy(Protocol):
    """Trainable policy that consumes a compact prefix observation."""

    def predict_action_batch(
        self,
        env_obs: PrefixObs,
        mode: str = "train",
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...


def extract_prefix_obs(
    feature_model: PrefixFeatureModel,
    env_obs: dict[str, Any],
) -> PrefixObs:
    """Extract a prefix observation through the canonical feature-model API."""
    extractor = getattr(feature_model, "extract_prefix_obs", None)
    if not callable(extractor):
        raise TypeError(
            f"{type(feature_model).__name__} must implement extract_prefix_obs()."
        )
    return extractor(env_obs)
