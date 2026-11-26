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

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from rlinf.models.embodiment.supermimic.supermimic_policy import SuperMimicPolicy


def get_model(cfg: DictConfig, torch_dtype=None) -> SuperMimicPolicy:
    """Build :class:`SuperMimicPolicy` from a model config."""
    del torch_dtype
    from rlinf.models.embodiment.supermimic.supermimic_policy import SuperMimicPolicy

    return SuperMimicPolicy.from_config(cfg)
