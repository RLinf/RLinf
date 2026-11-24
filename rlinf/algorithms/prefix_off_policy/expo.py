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

"""EXPO: select frozen proposals and bounded edits with target ensemble critics."""

import math
from typing import Any

import torch
import torch.nn.functional as F

from rlinf.algorithms.prefix_off_policy.base import (
    PrefixOffPolicyAlgorithm,
    PrefixOffPolicyContext,
    rollout_actor_target_critic,
)
from rlinf.algorithms.prefix_off_policy.capabilities import PrefixAlgorithmRequirements
from rlinf.algorithms.prefix_off_policy.registry import (
    register_prefix_off_policy_algorithm,
)
from rlinf.algorithms.prefix_off_policy.transition import (
    use_simulator_transition_replay,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.residual import expand_edit_scales


@register_prefix_off_policy_algorithm("expo")
class PrefixEXPOAlgorithm(PrefixOffPolicyAlgorithm):
    """Chunk-level EXPO objectives on the shared Prefix replay and learner runtime."""

    requirements = PrefixAlgorithmRequirements(
        actor_distribution="stochastic",
        trains_entropy_temperature=True,
        rollout_uses_target_critic=True,
    )

    def __init__(self, cfg: Any) -> None:
        super().__init__(cfg)
        self.num_candidates = cfg.algorithm.get("num_candidates", 8)
        self.num_edits = cfg.algorithm.get("num_edits", 8)
        if type(self.num_candidates) is not int or self.num_candidates < 1:
            raise ValueError("EXPO num_candidates must be a positive integer.")
        if (
            type(self.num_edits) is not int
            or not 0 <= self.num_edits <= self.num_candidates
        ):
            raise ValueError("EXPO requires 0 <= num_edits <= num_candidates.")
        if not self.actor_capabilities.bounded_residual:
            raise ValueError("EXPO requires a bounded residual actor head.")
        if self.critic_capabilities.supports_cross_q:
            raise ValueError("EXPO requires independent target critics, not CrossQ.")
        if use_simulator_transition_replay(cfg):
            raise ValueError(
                "EXPO requires algorithm.rlt_transition_replay=false for action chunks."
            )
        if cfg.algorithm.get("backup_entropy", False):
            raise ValueError("EXPO requires backup_entropy=false.")
        if cfg.algorithm.get("bootstrap_type", "standard") != "standard":
            raise ValueError("EXPO requires bootstrap_type=standard.")
        self._configured_actor_agg_q(default="mean", allowed=("mean",))
        if cfg.algorithm.get("reference_dropout_prob", 0):
            raise ValueError("EXPO requires reference_dropout_prob=0.")
        entropy = cfg.algorithm.entropy_tuning
        alpha_type = entropy.get("alpha_type", "exp")
        initial = float(entropy.get("initial_alpha", 1e-5))
        if (
            alpha_type not in {"exp", "softplus", "fixed_alpha"}
            or not math.isfinite(initial)
            or initial < 0
            or (alpha_type != "fixed_alpha" and initial == 0)
        ):
            raise ValueError(
                "EXPO requires finite positive trainable alpha or nonnegative fixed alpha."
            )
        if (
            cfg.algorithm.get("bc_weight", 0.0)
            or cfg.algorithm.get("q_weight", 1.0) != 1.0
            or cfg.algorithm.get("actor_weight_schedule", {}).get("enable", False)
        ):
            raise ValueError(
                "EXPO uses its entropy/Q objective without BC or actor weight schedules."
            )
        self.target_entropy = self.resolve_target_entropy()

    def rollout_state_dict(self, online: dict, target: dict) -> dict:
        """Combine the online editor and target critic for candidate selection."""
        return rollout_actor_target_critic(online, target)

    def resolve_target_entropy(self) -> float:
        """Measure entropy over every residual coordinate in the executed chunk."""
        chunk, width = self._chunk_shape()
        scales = expand_edit_scales(
            self.model_cfg.actor_head.get("edit_scale", 0.02), chunk * width
        )
        supplied = self.cfg.algorithm.entropy_tuning.get("target_entropy")
        target = (
            sum(math.log(scale) - 0.5 for scale in scales)
            if supplied is None
            else float(supplied)
        )
        if not math.isfinite(target) or target > sum(
            math.log(2 * scale) for scale in scales
        ):
            raise ValueError(
                "target_entropy must be finite and not exceed the residual support entropy."
            )
        return target

    def _sample_pair(self, device: torch.device, generator=None) -> torch.Tensor:
        return torch.randperm(self.num_q_heads, device=device, generator=generator)[:2]

    @torch.no_grad()
    def select_actions(
        self, actor: Any, critic: Any, obs: dict, *, generator=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank K frozen proposals and M online edits by a random-pair target minimum."""
        chunk, width = self._chunk_shape()
        candidates = obs["candidates"]
        expected = (self.num_candidates, chunk, width)
        if candidates.ndim != 4 or tuple(candidates.shape[1:]) != expected:
            raise ValueError(
                f"Expected candidates [B, {expected}], got {tuple(candidates.shape)}."
            )
        batch = candidates.shape[0]
        candidates = candidates.flatten(2)
        if self.num_edits:
            edit_obs = {
                key: obs[key].repeat_interleave(self.num_edits, dim=0)
                for key in ("z_rl", "proprio")
            }
            edit_obs["ref_chunk"] = candidates[:, : self.num_edits].reshape(
                batch * self.num_edits, chunk, width
            )
            edited, _, _ = actor(forward_type=ForwardType.SAC, obs=edit_obs)
            candidates = torch.cat(
                (candidates, edited.reshape(batch, self.num_edits, -1)), 1
            )
        count = candidates.shape[1]
        score_obs = {
            key: obs[key].repeat_interleave(count, dim=0) for key in ("z_rl", "proprio")
        }
        values = critic(
            forward_type=ForwardType.SAC_Q,
            obs=score_obs,
            actions=candidates.flatten(0, 1),
        )
        pair = self._sample_pair(candidates.device, generator)
        scores = values.index_select(-1, pair).amin(-1).reshape(batch, count)
        indices = scores.argmax(-1)
        return candidates[
            torch.arange(batch, device=candidates.device), indices
        ], indices

    def target_actions(
        self, context: PrefixOffPolicyContext, next_obs: dict
    ) -> torch.Tensor:
        """Use the online editor and target critics for next-state selection."""
        return self.select_actions(
            context.model,
            context.target_model,
            next_obs,
            generator=context.critic_sample_generator,
        )[0]

    def critic_loss(
        self, context: PrefixOffPolicyContext, batch: dict
    ) -> tuple[torch.Tensor, dict]:
        """Sum chunk rewards and bootstrap once, without entropy or time-limit termination."""
        generator = context.critic_sample_generator
        with torch.no_grad():
            action, indices = self.select_actions(
                context.model,
                context.target_model,
                batch["next_obs"],
                generator=generator,
            )
            next_values = context.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=batch["next_obs"],
                actions=action,
            )
            # Selection and backup need independent critic pairs.
            pair = self._sample_pair(action.device, generator)
            next_q = next_values.index_select(-1, pair).amin(-1, keepdim=True)
            reward = (
                batch["rewards"]
                .float()
                .reshape(action.shape[0], -1)
                .sum(-1, keepdim=True)
            )
            terminal = (
                batch["terminations"]
                .bool()
                .reshape(action.shape[0], -1)
                .any(-1, keepdim=True)
            )
            if not self.cfg.algorithm.get("bootstrap_time_limits", True):
                terminal |= (
                    batch["truncations"]
                    .bool()
                    .reshape(action.shape[0], -1)
                    .any(-1, keepdim=True)
                )
            gamma = (
                context._env_values("gamma", batch["env_index"], torch.float32)
                if "env_index" in batch
                else self.cfg.algorithm.gamma
            )
            target = reward + gamma * (~terminal) * next_q
        values = context.model(
            forward_type=ForwardType.SAC_Q,
            obs=batch["curr_obs"],
            actions=batch["actions"],
        )
        loss = F.mse_loss(values, target.expand_as(values))
        return loss, {
            "q_data": values.detach().mean().item(),
            "target_q": target.mean().item(),
            "td_error_abs": (values.detach() - target).abs().mean().item(),
            "target_edit_fraction": (indices >= self.num_candidates)
            .float()
            .mean()
            .item(),
        }

    def _edit_replay_action(
        self, context: PrefixOffPolicyContext, batch: dict
    ) -> tuple:
        obs = {**batch["curr_obs"], "ref_chunk": batch["actions"]}
        action, log_prob, _ = context.model(forward_type=ForwardType.SAC, obs=obs)
        return obs, action, log_prob.sum(-1, keepdim=True)

    def actor_loss(self, context: PrefixOffPolicyContext, batch: dict) -> tuple:
        """Edit executed replay actions; maximize the online ensemble mean minus entropy cost."""
        obs, action, log_prob = self._edit_replay_action(context, batch)
        values = context.model(
            forward_type=ForwardType.SAC_Q,
            obs=obs,
            actions=action,
            detach_encoder=True,
        )
        q = values.mean(-1, keepdim=True)
        alpha = context.entropy_temp.compute_alpha().detach()
        loss = (alpha * log_prob - q).mean()
        return (
            loss,
            -log_prob.mean(),
            {
                "q_pi": q.detach().mean().item(),
                "weighted_entropy": (alpha * log_prob).detach().mean().item(),
            },
        )

    def alpha_loss(self, context: PrefixOffPolicyContext, batch: dict) -> torch.Tensor:
        """Fit temperature using the same executed-action editor conditioning as the actor."""
        with torch.no_grad():
            _, _, log_prob = self._edit_replay_action(context, batch)
        return -context.entropy_temp.compute_alpha() * (
            log_prob.mean() + self.target_entropy
        )

    @torch.no_grad()
    def predict_action_batch(self, model: Any, obs: dict, *, mode: str) -> tuple:
        """Select stochastically in train and eval; rollout Q weights are target weights."""
        obs = model.preprocess_env_obs(obs)
        action, indices = self.select_actions(model, model, obs)
        chunk, width = self._chunk_shape()
        return action.reshape(-1, chunk, width), {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                **obs,
                "action": action,
                "model_action": action,
                "expo_selected_index": indices,
            },
        }
