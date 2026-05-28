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

"""LingBot-VA embodied model integration for RLinf (Libero evaluation)."""

from __future__ import annotations

import torch
from omegaconf import DictConfig

from rlinf.utils.logging import get_logger


def get_model(cfg: DictConfig, torch_dtype: torch.dtype | None = None):
    """Build the LingBot-VA model adapter for Libero evaluation."""
    from rlinf.models.embodiment.lingbotva.lingbotva_action_model import (
        LingbotVAActionModel,
    )

    model = LingbotVAActionModel(cfg=cfg, torch_dtype=torch_dtype or torch.bfloat16)
    get_logger().info("Initialized LingBot-VA adapter for Libero.")
    return model
