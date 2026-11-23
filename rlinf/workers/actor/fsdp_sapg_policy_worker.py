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

"""Leader-follower SAPG updates for embodied training."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.algorithms.sapg import (
    relabel_sapg_batch,
    sample_repeat_idxs,
    sapg_block_ids,
    sapg_entropy_coef_for_states,
    sapg_follower_indices,
)
from rlinf.utils.distributed import masked_stats
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedSAPGFSDPActor(EmbodiedFSDPActor):
    """Train every block on its own samples and aggregate into the leader."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.sapg_cfg = cfg.actor.model.sapg
        self._sapg_num_blocks = int(self.sapg_cfg.get("num_blocks", 1))
        self._sapg_embd_size = int(self.sapg_cfg.get("embd_size", 16))
        self._sapg_off_policy_ratio = int(self.sapg_cfg.get("off_policy_ratio", 1))
        if not self.sapg_cfg.get("leader_aggregate", True):
            self._sapg_off_policy_ratio = 0

    def needs_entropy_for_loss(self) -> bool:
        return (
            self.sapg_cfg.get("intrinsic_reward_type", "entropy") == "entropy"
            and float(self.sapg_cfg.get("intrinsic_coef_scale", 1.0)) != 0.0
        )

    def extra_loss_kwargs(self, micro_batch, output_dict) -> dict:
        entropy = output_dict.get("entropy")
        coef = None
        if self.needs_entropy_for_loss():
            coef = sapg_entropy_coef_for_states(
                micro_batch["forward_inputs"]["states"],
                num_blocks=self._sapg_num_blocks,
                embd_size=self._sapg_embd_size,
                intrinsic_coef_scale=float(
                    self.sapg_cfg.get("intrinsic_coef_scale", 1.0)
                ),
                intrinsic_coef_start=float(
                    self.sapg_cfg.get("intrinsic_coef_start", 0.5)
                ),
                intrinsic_coef_end=float(self.sapg_cfg.get("intrinsic_coef_end", 0.0)),
                embedding_start=float(self.sapg_cfg.get("embedding_start", 50.0)),
                embedding_end=float(self.sapg_cfg.get("embedding_end", 0.0)),
            )
        return {
            "entropy": entropy,
            "sapg_entropy_coef": coef,
            "critic_coef": self.cfg.algorithm.get("critic_coef", 4.0),
            "action_means": output_dict["action_means"],
            "bounds_loss_coef": self.cfg.algorithm.get("bounds_loss_coef", 0.0001),
        }

    @torch.no_grad()
    def _compute_rollout_advantages(self, batch: dict[str, Any], **kwargs) -> None:
        normalize_advantages = kwargs.pop("normalize_advantages", True)
        super()._compute_rollout_advantages(batch, **kwargs, normalize_advantages=False)
        if self._sapg_off_policy_ratio and self._sapg_num_blocks > 1:
            self._aggregate_follower_transitions(batch, kwargs["gamma"])
        if normalize_advantages:
            self._normalize_sapg_advantages(batch)

    def _normalize_advantages_jointly(self) -> None:
        (batch,) = self.env_batches.values()
        self._normalize_sapg_advantages(batch)

    def _normalize_sapg_advantages(self, batch: dict[str, Any]) -> None:
        advantages = batch["advantages"]
        stats = masked_stats(advantages, batch.get("loss_mask")).to(self.device)
        torch.distributed.all_reduce(stats)
        count, total, squared_total = stats.to(advantages.device).unbind()
        count = count.clamp_min(1)
        mean = total / count
        variance = (
            (squared_total - total * mean) / (count - 1).clamp_min(1)
        ).clamp_min(0)
        batch["advantages"] = (advantages - mean.to(advantages)) / (
            variance.sqrt().to(advantages) + 1e-8
        )

    def _aggregate_follower_transitions(self, batch: dict, gamma: float) -> None:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        codes = self.model.block_codes.detach().cpu()
        states = batch["forward_inputs"]["states"]
        block_ids = sapg_block_ids(states[0], codes)
        generator = torch.Generator().manual_seed(self.cfg.actor.seed + self.version)
        shifts = sample_repeat_idxs(
            self._sapg_num_blocks, self._sapg_off_policy_ratio, generator
        )
        followers = sapg_follower_indices(block_ids, self._sapg_num_blocks, shifts)
        relabel_sapg_batch(
            batch, followers, codes[-1], gamma, self._score_leader_values
        )

    def _score_leader_values(self, states: torch.Tensor) -> torch.Tensor:
        flat = states.flatten(0, 1)
        values = []
        for chunk in flat.split(self.cfg.actor.micro_batch_size):
            with self.amp_context:
                output = self.model(
                    forward_inputs=put_tensor_device({"states": chunk}, self.device),
                    compute_logprobs=False,
                    compute_entropy=False,
                    compute_values=True,
                )
            values.append(output["values"].float().cpu())
        return torch.cat(values, dim=0).reshape(*states.shape[:2], -1)
