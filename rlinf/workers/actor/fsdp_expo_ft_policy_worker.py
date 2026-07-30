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
import torch.nn.functional as F

from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    RLTACFSDPPolicy,
    RLTACLossMixin,
)
from rlinf.workers.actor.fsdp_rlt_td3_policy_worker import (
    RLTTD3FSDPPolicy,
    RLTTD3LossMixin,
)


class ExpoFTACLossMixin:
    """EXPO-FT losses while reusing RLT replay, routing, and scheduling."""

    def _use_td3_mlp_losses(self) -> bool:
        return self.cfg.actor.model.get("model_type") == "rlt_td3_mlp_policy"

    @staticmethod
    def _flatten_chunk(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _num_q_heads(self) -> int:
        return int(self.cfg.actor.model.get("num_q_heads", 10))

    def _num_min_qs(self) -> int:
        return int(self.cfg.algorithm.get("num_min_qs", 2))

    def _q_indices(self, *, mode: str) -> torch.Tensor | None:
        num_q_heads = self._num_q_heads()
        num_min_qs = self._num_min_qs()
        if num_min_qs <= 0 or num_min_qs > num_q_heads:
            raise ValueError(
                f"num_min_qs must be in [1, {num_q_heads}], got {num_min_qs}."
            )
        if num_min_qs == num_q_heads:
            return None
        if mode == "eval":
            return torch.arange(num_min_qs, device=self.device)
        return torch.randperm(num_q_heads, device=self.device)[:num_min_qs]

    @staticmethod
    def _min_q(q_values: torch.Tensor) -> torch.Tensor:
        if q_values.shape[-1] < 1:
            raise ValueError(f"Expected at least one Q head, got {q_values.shape}.")
        return q_values.min(dim=-1, keepdim=True).values

    def _actor_q(self, q_values: torch.Tensor) -> torch.Tensor:
        agg_q = self.cfg.algorithm.get("actor_agg_q", "mean")
        if agg_q == "mean":
            return q_values.mean(dim=-1, keepdim=True)
        if agg_q == "min":
            return self._min_q(q_values)
        if agg_q == "q1":
            return q_values[..., :1]
        raise NotImplementedError(f"{agg_q=} is not supported for EXPO-FT.")

    def _discounted_chunk_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.reshape(rewards.shape[0], -1).to(self.torch_dtype)
        chunk_len = rewards.shape[-1]
        discounts = torch.pow(
            torch.as_tensor(self.cfg.algorithm.gamma, device=rewards.device),
            torch.arange(chunk_len, device=rewards.device, dtype=rewards.dtype),
        )
        return torch.sum(rewards * discounts, dim=-1, keepdim=True)

    def _base_actions_for_actor(self, batch: dict) -> torch.Tensor:
        source = self.cfg.algorithm.get("residual_actor_base_source", "ref_chunk")
        if source == "data_action":
            return self._flatten_chunk(batch["actions"])
        if source == "ref_chunk":
            ref_chunk = batch["curr_obs"]["ref_chunk"]
            action_dim = int(self.cfg.actor.model.action_dim)
            chunk_len = int(self.cfg.actor.model.num_action_chunks)
            if ref_chunk.dim() == 2:
                ref_chunk = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)
            return ref_chunk[:, :chunk_len].reshape(ref_chunk.shape[0], -1)
        raise NotImplementedError(
            f"{source=} is not supported for residual_actor_base_source."
        )

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        if self._use_td3_mlp_losses():
            return RLTACLossMixin.forward_critic(self, batch)

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        done_source = (
            batch["dones"]
            if use_simulator_transition_replay(self.cfg)
            else batch["terminations"]
        )
        not_done = ~done_source.reshape(done_source.shape[0], -1).bool().any(
            dim=-1, keepdim=True
        )

        with torch.no_grad():
            q_indices = self._q_indices(mode="train")
            next_actions, _, selection_info = self.target_model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
                select_top_q=True,
                q_indices=q_indices,
            )
            all_qf_next_target = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=next_obs,
                actions=next_actions,
            )
            if q_indices is not None:
                all_qf_next_target = all_qf_next_target.index_select(
                    dim=-1, index=q_indices
                )
            q_next = self._min_q(all_qf_next_target)

            reward_target = self._discounted_chunk_rewards(rewards)
            reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
            bootstrap_discount = self.cfg.algorithm.gamma**reward_horizon
            bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
            if bootstrap_type == "always":
                target_q_values = reward_target + bootstrap_discount * q_next
            elif bootstrap_type == "standard":
                target_q_values = reward_target + not_done * bootstrap_discount * q_next
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported.")

        all_data_q_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
        )
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        selected_indices = selection_info["selected_indices"]
        num_base_candidates = int(self.cfg.actor.model.get("num_base_candidates", 8))
        metrics = {
            "q_data": all_data_q_values.mean().item(),
            "q_next": q_next.mean().item(),
            "target_q": target_q_values.mean().item(),
            "vf_select_ratio_base": (selected_indices < num_base_candidates)
            .float()
            .mean()
            .item(),
            "vf_select_ratio_residual": (selected_indices >= num_base_candidates)
            .float()
            .mean()
            .item(),
        }
        return critic_loss, metrics

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        if self._use_td3_mlp_losses():
            return RLTTD3LossMixin.forward_actor(self, batch)

        curr_obs = batch["curr_obs"]
        base_actions = self._base_actions_for_actor(batch)
        pi, log_pi, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            base_actions=base_actions,
        )
        log_pi = log_pi.reshape(log_pi.shape[0], -1).sum(dim=-1, keepdim=True)

        all_qf_pi = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=pi,
            detach_encoder=True,
        )
        qf_pi = self._actor_q(all_qf_pi)

        alpha = self.entropy_temp.compute_alpha().to(dtype=log_pi.dtype)
        q_weight = float(self.cfg.algorithm.get("q_weight", 1.0))
        actor_loss = (alpha * log_pi - q_weight * qf_pi).mean()
        entropy = -log_pi.mean()

        metrics = {
            "q_pi": qf_pi.mean().item(),
            "entropy": entropy.item(),
            "alpha": alpha.detach().mean().item(),
            "action_edit_abs_mean": (pi - base_actions).abs().mean().item(),
        }
        for q_id in range(all_qf_pi.shape[-1]):
            metrics[f"q_value_{q_id}"] = all_qf_pi[..., q_id].mean().item()
        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        if self._use_td3_mlp_losses():
            return RLTACLossMixin.forward_alpha(self, batch)

        curr_obs = batch["curr_obs"]
        base_actions = self._base_actions_for_actor(batch)
        with torch.no_grad():
            _, log_pi, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=curr_obs,
                base_actions=base_actions,
            )
            log_pi = log_pi.reshape(log_pi.shape[0], -1).sum(dim=-1, keepdim=True)

        alpha = self.entropy_temp.compute_alpha()
        return -alpha * (log_pi.mean() + self.target_entropy)


class ExpoFTACFSDPPolicy(ExpoFTACLossMixin, RLTTD3LossMixin, RLTACFSDPPolicy):
    """Synchronous EXPO-FT worker with RLT transition replay."""

    def update_one_epoch(self, train_actor: bool = True):
        if self._use_td3_mlp_losses():
            return RLTTD3FSDPPolicy.update_one_epoch(
                self,
                train_actor=train_actor,
            )
        return super().update_one_epoch(train_actor=train_actor)
