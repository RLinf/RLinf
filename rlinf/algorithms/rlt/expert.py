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

import copy
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict


def build_expert_model_config(
    cfg: Any,
    model_cfg: Any,
    *,
    rlt_feature_model_config: Any | None = None,
):
    """Build rollout expert config for DAgger or ManiSkill RLT."""
    expert_cfg = cfg.rollout.expert_model
    if rlt_feature_model_config is not None:
        expert_overrides = OmegaConf.to_container(expert_cfg, resolve=True)
        expert_overrides = {} if expert_overrides is None else dict(expert_overrides)
        return OmegaConf.merge(
            copy.deepcopy(rlt_feature_model_config), expert_overrides
        )

    expert_model_config = copy.deepcopy(model_cfg)
    with open_dict(expert_model_config):
        expert_model_config.precision = expert_cfg.precision
        expert_model_config.model_path = expert_cfg.model_path
    return expert_model_config


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
