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

from rlinf.models.embodiment.prefix_ft.types import PrefixObs


@runtime_checkable
class PrefixFeatureModel(Protocol):
    """Frozen VLA that exposes a pooled prefix plus reference actions."""

    @property
    def z_dim(self) -> int: ...

    def extract_prefix_obs(self, env_obs: dict[str, Any]) -> PrefixObs:
        """One forward: prefix hidden -> pool -> z; sample ref_chunk; slice proprio."""
        ...


def extract_prefix_obs(feature_model: Any, env_obs: dict[str, Any]) -> PrefixObs:
    """Call ``extract_prefix_obs`` if present, else the ``extract_rlt_obs`` alias."""
    extractor = getattr(feature_model, "extract_prefix_obs", None)
    if callable(extractor):
        return extractor(env_obs)
    extractor = getattr(feature_model, "extract_rlt_obs", None)
    if callable(extractor):
        return extractor(env_obs)
    raise TypeError(
        f"{type(feature_model).__name__} does not implement extract_prefix_obs "
        "or extract_rlt_obs."
    )


@runtime_checkable
class PrefixPolicy(Protocol):
    """Trainable head that consumes PrefixObs, not raw images."""

    def predict_action_batch(
        self,
        env_obs: PrefixObs,
        mode: str = "train",
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]: ...
