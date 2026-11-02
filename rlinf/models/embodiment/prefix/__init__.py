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

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.prefix.config import (
    build_state_history_buffer,
    get_prefix_feature_dim,
    get_prefix_feature_model_config,
    get_state_history_config,
    is_prefix_off_policy_loss,
    resolve_prefix_pool,
)
from rlinf.models.embodiment.prefix.contracts import (
    PrefixFeatureModel,
    TrainablePrefixPolicy,
    extract_prefix_obs,
)
from rlinf.models.embodiment.prefix.heads import (
    register_prefix_actor,
    register_prefix_critic,
)
from rlinf.models.embodiment.prefix.history import StateHistoryBuffer
from rlinf.models.embodiment.prefix.policy import PrefixPolicy
from rlinf.models.embodiment.prefix.pool import pool_prefix
from rlinf.models.embodiment.prefix.types import PREFIX_OBS_KEYS, PrefixObs


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16) -> PrefixPolicy:
    """Build a prefix policy from registered actor and critic heads."""
    del torch_dtype
    return PrefixPolicy(
        prefix_dim=get_prefix_feature_dim(cfg),
        proprio_dim=cfg.proprio_dim,
        action_dim=cfg.action_dim,
        num_action_chunks=cfg.num_action_chunks,
        ref_num_action_chunks=cfg.get(
            "ref_num_action_chunks",
            cfg.num_action_chunks,
        ),
        actor_head_cfg=cfg.actor_head,
        critic_head_cfg=cfg.critic_head,
    )


__all__ = [
    "PREFIX_OBS_KEYS",
    "PrefixFeatureModel",
    "PrefixObs",
    "PrefixPolicy",
    "StateHistoryBuffer",
    "TrainablePrefixPolicy",
    "build_state_history_buffer",
    "extract_prefix_obs",
    "get_model",
    "get_prefix_feature_dim",
    "get_prefix_feature_model_config",
    "get_state_history_config",
    "is_prefix_off_policy_loss",
    "pool_prefix",
    "register_prefix_actor",
    "register_prefix_critic",
    "resolve_prefix_pool",
]
