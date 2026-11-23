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
import torch.nn as nn
from omegaconf import DictConfig

from rlinf.algorithms.sapg import build_sapg_exploration, sapg_block_ids
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


class SAPGMLPPolicy(MLPPolicy):
    """Shared networks with learned exploration embeddings and per-block log std."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        num_action_chunks: int,
        add_value_head: bool,
        sapg_cfg: DictConfig,
        add_q_head: bool = False,
        q_head_type: str = "default",
    ):
        self.raw_obs_dim = obs_dim
        self.embd_size = int(sapg_cfg.get("embd_size", 16))
        self.num_blocks = int(sapg_cfg.get("num_blocks", 1))
        super().__init__(
            obs_dim=obs_dim + self.embd_size,
            action_dim=action_dim,
            num_action_chunks=num_action_chunks,
            add_value_head=add_value_head,
            add_q_head=add_q_head,
            q_head_type=q_head_type,
        )
        spec = build_sapg_exploration(
            num_envs=self.num_blocks,
            num_blocks=self.num_blocks,
            embd_size=self.embd_size,
            embedding_start=sapg_cfg.get("embedding_start", 50.0),
            embedding_end=sapg_cfg.get("embedding_end", 0.0),
        )
        self.register_buffer("block_codes", spec.embedding, persistent=False)
        self.block_embeddings = nn.Parameter(torch.randn_like(spec.embedding))
        self.actor_logstd = nn.Parameter(
            torch.zeros(self.num_blocks, num_action_chunks * action_dim)
        )

    def _conditioned_states(
        self, states: torch.Tensor, block_ids: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            [states[..., : self.raw_obs_dim], self.block_embeddings[block_ids]], dim=-1
        )

    def _action_parameters(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block_ids = sapg_block_ids(states, self.block_codes)
        features = self.backbone(self._conditioned_states(states, block_ids))
        return self.actor_mean(features), self.actor_logstd[block_ids]

    def _predict_values(self, states: torch.Tensor) -> torch.Tensor:
        block_ids = sapg_block_ids(states, self.block_codes)
        return super()._predict_values(self._conditioned_states(states, block_ids))

    def default_forward(self, *args, **kwargs):
        kwargs.setdefault("compute_action_means", kwargs.get("compute_logprobs", True))
        return super().default_forward(*args, **kwargs)

    def preprocess_env_obs(self, env_obs: dict, mode: str = "train") -> dict:
        """Attach fixed block codes for rollout routing and replay."""
        raw = env_obs["states"].to(self.block_codes.device)
        if mode == "eval":
            codes = self.block_codes[-1:].expand(raw.shape[0], -1)
        else:
            if raw.shape[0] % self.num_blocks:
                raise ValueError(
                    f"SAPG training batch size {raw.shape[0]} must be divisible "
                    f"by the number of blocks ({self.num_blocks})."
                )
            # RLinf routes contiguous slices of the rollout batch to actors.
            # Interleave blocks so each aligned slice sees every policy.
            codes = self.block_codes.repeat(raw.shape[0] // self.num_blocks, 1)
        return {"states": torch.cat([raw, codes.to(raw)], dim=-1)}

    def _preprocess_action_obs(self, env_obs, mode):
        return self.preprocess_env_obs(env_obs, mode=mode)
