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

"""EXPO-style edit actor training for Prefix policies."""

from typing import Any

import torch
import torch.nn.functional as F

from rlinf.algorithms.prefix_off_policy.base import (
    PrefixOffPolicyAlgorithm,
    PrefixOffPolicyContext,
)
from rlinf.algorithms.prefix_off_policy.capabilities import (
    PrefixAlgorithmRequirements,
)
from rlinf.algorithms.prefix_off_policy.registry import (
    register_prefix_off_policy_algorithm,
)
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.models.embodiment.base_policy import ForwardType


@register_prefix_off_policy_algorithm("expo")
class PrefixEXPOAlgorithm(PrefixOffPolicyAlgorithm):
    """Train a stochastic edit policy around the frozen VLA reference chunk.

    The reference action is always included in the target candidate set. The
    edit actor supplies additional candidates, and the target critic selects
    the highest-value candidate for the chunk-level TD backup.
    """

    requirements = PrefixAlgorithmRequirements(
        actor_distribution="stochastic",
        min_q_heads=2,
    )

    def validate_configuration(self) -> None:
        super().validate_configuration()
        num_samples = int(self.cfg.algorithm.get("expo_num_edit_samples", 8))
        if num_samples < 0:
            raise ValueError("algorithm.expo_num_edit_samples must be >= 0.")
        num_min_qs = int(self.cfg.algorithm.get("expo_num_min_qs", 2))
        if not 1 <= num_min_qs <= self.num_q_heads:
            raise ValueError(
                "algorithm.expo_num_min_qs must be between 1 and "
                f"critic_head.num_q_heads ({self.num_q_heads})."
            )

    def _num_edit_samples(self) -> int:
        return int(self.cfg.algorithm.get("expo_num_edit_samples", 8))

    def _num_min_qs(self, num_q_heads: int) -> int:
        return int(self.cfg.algorithm.get("expo_num_min_qs", 2))

    @staticmethod
    def _repeat_obs(
        obs: dict[str, torch.Tensor], repeats: int
    ) -> dict[str, torch.Tensor]:
        if repeats == 1:
            return obs
        return {
            key: value.repeat_interleave(repeats, dim=0) for key, value in obs.items()
        }

    def _q_score(self, q_values: torch.Tensor) -> torch.Tensor:
        """Return a REDQ-style minimum over a random Q subset."""
        num_q_heads = q_values.shape[-1]
        num_min_qs = self._num_min_qs(num_q_heads)
        if num_min_qs == num_q_heads:
            return q_values.min(dim=-1).values
        indices = torch.randperm(
            num_q_heads,
            device=q_values.device,
        )[:num_min_qs]
        return q_values.index_select(-1, indices).min(dim=-1).values

    def _candidate_actions(
        self,
        model: Any,
        obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        reference = self._ref_chunk(obs)
        num_samples = self._num_edit_samples()
        if num_samples == 0:
            return reference.unsqueeze(1)
        repeated_obs = self._repeat_obs(obs, num_samples)
        edits, _, _ = model(
            forward_type=ForwardType.SAC,
            obs=repeated_obs,
            deterministic=False,
            apply_reference_dropout=False,
            apply_action_noise=False,
        )
        edits = edits.reshape(reference.shape[0], num_samples, -1)
        return torch.cat([reference.unsqueeze(1), edits], dim=1)

    def _candidate_q_values(
        self,
        model: Any,
        obs: dict[str, torch.Tensor],
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_candidates, action_dim = candidates.shape
        repeated_obs = self._repeat_obs(obs, num_candidates)
        q_values = model(
            forward_type=ForwardType.SAC_Q,
            obs=repeated_obs,
            actions=candidates.reshape(batch_size * num_candidates, action_dim),
        )
        if isinstance(q_values, tuple):
            q_values = q_values[0]
        return q_values.reshape(batch_size, num_candidates, -1)

    def actor_forward_kwargs(self) -> dict[str, object]:
        return {
            "apply_reference_dropout": False,
            "reference_dropout_prob": 0.0,
            "apply_action_noise": False,
        }

    def target_actions(
        self,
        context: PrefixOffPolicyContext,
        next_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        candidates = self._candidate_actions(context.target_model, next_obs)
        candidate_q = self._candidate_q_values(
            context.target_model,
            next_obs,
            candidates,
        )
        selected = self._q_score(candidate_q).argmax(dim=1)
        rows = torch.arange(candidates.shape[0], device=candidates.device)
        return candidates[rows, selected]

    def aggregate_actor_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        return all_q_values.mean(dim=-1, keepdim=True)

    def critic_loss(
        self,
        context: PrefixOffPolicyContext,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        done_source = batch["terminations"]
        if use_simulator_transition_replay(self.cfg):
            done_source = batch["dones"]
        not_done = ~done_source.reshape(done_source.shape[0], -1).bool().any(
            dim=-1,
            keepdim=True,
        )

        with torch.no_grad():
            candidates = self._candidate_actions(context.target_model, next_obs)
            candidate_q = self._candidate_q_values(
                context.target_model,
                next_obs,
                candidates,
            )
            q_next = self._q_score(candidate_q).max(dim=1, keepdim=True).values
            reward_target = self._discounted_chunk_rewards(
                rewards,
                context.torch_dtype,
            )
            reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
            bootstrap_discount = self.cfg.algorithm.gamma**reward_horizon
            target_q_values = reward_target + (not_done * bootstrap_discount * q_next)

        all_data_q_values = context.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
        )
        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values,
            target_q_values.expand_as(all_data_q_values),
        )
        return critic_loss, {
            "q_data": all_data_q_values.mean().item(),
            "q_target": target_q_values.mean().item(),
            "expo_candidates": float(candidates.shape[1]),
        }

    def actor_loss(
        self,
        context: PrefixOffPolicyContext,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        curr_obs = batch["curr_obs"]
        pi, log_pi, _ = context.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            **self.actor_forward_kwargs(),
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)

        all_q_values = context.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=pi,
            detach_encoder=True,
        )
        q_pi = self.aggregate_actor_q(all_q_values)
        alpha = float(context.entropy_temp.alpha)
        bc_weight, q_weight, weight_metrics = self._actor_objective_weights(
            int(context.update_step)
        )
        ref_chunk = self._ref_chunk(curr_obs)
        bc_loss, bc_metrics = self._bc_metrics(
            pi=pi,
            actions=batch["actions"],
            ref_chunk=ref_chunk,
            intervene_flags=batch.get("intervene_flags"),
        )
        actor_loss = (
            alpha * log_pi.mean() - q_weight * q_pi.mean() + bc_weight * bc_loss
        )
        metrics = {
            **bc_metrics,
            **weight_metrics,
            "q_pi": q_pi.mean().item(),
            "alpha": alpha,
            "edit_abs_mean": (pi - ref_chunk).abs().mean().item(),
            "weighted_q": (q_weight * q_pi.mean()).detach().item(),
            "weighted_bc": (bc_weight * bc_loss).detach().item(),
        }
        return actor_loss, -log_pi.mean(), metrics

    def alpha_loss(
        self,
        context: PrefixOffPolicyContext,
        batch: dict[str, Any],
    ) -> torch.Tensor:
        with torch.no_grad():
            _, log_pi, _ = context.model(
                forward_type=ForwardType.SAC,
                obs=batch["curr_obs"],
                **self.actor_forward_kwargs(),
            )
            if log_pi.ndim == 1:
                log_pi = log_pi.unsqueeze(-1)
            log_pi = log_pi.sum(dim=-1, keepdim=True)

        entropy_cfg = self.cfg.algorithm.get("entropy_tuning", {}) or {}
        _, action_dim = self._chunk_shape()
        full_action_dim = int(self.cfg.actor.model.num_action_chunks) * action_dim
        target_entropy = float(
            entropy_cfg.get(
                "target_entropy",
                -full_action_dim / 2.0,
            )
        )
        alpha = context.entropy_temp.compute_alpha()
        return -alpha * (log_pi.mean() + target_entropy)
