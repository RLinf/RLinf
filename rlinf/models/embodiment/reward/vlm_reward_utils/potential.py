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

"""Scalar potential-head helpers for VLM Trend Success + Potential rewards."""

from __future__ import annotations

import torch


class ScalarPotentialHead(torch.nn.Module):
    """Map frozen VLM prompt features to scalar potential logits."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(input_dim),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map features of shape ``(batch, dim)`` to scalar logits."""
        return self.net(features).squeeze(-1)


@torch.no_grad()
def extract_prompt_features(
    model: torch.nn.Module,
    batched_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Pool the last attended token from the model's final hidden layer."""
    outputs = model(
        **batched_inputs,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.hidden_states[-1]
    attention_mask = batched_inputs["attention_mask"].bool()
    positions = torch.arange(
        attention_mask.shape[1], device=attention_mask.device
    ).unsqueeze(0)
    last_positions = positions.masked_fill(~attention_mask, -1).amax(dim=1)
    batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[batch_indices, last_positions].float()
