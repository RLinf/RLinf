# Copyright 2025 The RLinf Authors.
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

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    model = MLPPolicy(
        cfg.obs_dim,
        cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        add_value_head=cfg.add_value_head,
        add_q_head=cfg.get("add_q_head", False),
        q_head_type=cfg.get("q_head_type", "default"),
    )

    return model
