# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn

from rlinf.models.embodiment.openpi_pytorch.pi0_model.pi0 import Pi0


class OpenPiPytorchActionModel(nn.Module):
    """Abstract base wrapper around the vendored ``Pi0`` model.

    Concrete subclasses must provide their own ``predict_action_batch`` and
    ``forward`` (if training is needed). This base only wires up the Pi0
    model, the device shortcut, and the gradient-checkpointing pass-through.
    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
    ):
        super().__init__()
        self.model = pi0_model
        self.num_steps = num_steps
        self.action_env_dim = action_env_dim

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # --- Gradient checkpointing pass-through (used by the FSDP training path) ---
    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict | None = None, **kwargs
    ) -> None:
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self, **kwargs) -> None:
        self.model.gradient_checkpointing_disable()
