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

from typing import Any

import numpy as np
import torch

from rlinf.algorithms.expert import build_expert_model_config

__all__ = ["build_expert_model_config", "predict_expert_actions"]


def predict_expert_actions(
    expert_model: Any,
    env_obs: dict[str, Any],
    *,
    chunk_len: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    with torch.no_grad():
        expert_actions, _ = expert_model.predict_action_batch(
            env_obs=env_obs,
            mode="eval",
            compute_values=False,
        )
    if isinstance(expert_actions, np.ndarray):
        expert_actions = torch.from_numpy(expert_actions)
    if expert_actions.dim() == 2:
        expert_actions = expert_actions.reshape(expert_actions.shape[0], -1, action_dim)
    return expert_actions[:, :chunk_len, :action_dim].to(device=device, dtype=dtype)
