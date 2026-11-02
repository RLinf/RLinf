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

from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from rlinf.algorithms.prefix_off_policy.capabilities import (
    PrefixAlgorithmRequirements,
)
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.prefix.heads.registry import (
    get_prefix_actor_spec,
    get_prefix_critic_spec,
)


class PrefixOffPolicyContext(Protocol):
    """Worker-owned state needed by a prefix off-policy algorithm."""

    cfg: Any
    model: Any
    target_model: Any
    torch_dtype: torch.dtype
    update_step: int


class PrefixOffPolicyAlgorithm(ABC):
    """Algorithm contract shared by prefix AC and TD3 variants."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        actor_cfg = cfg.actor.model.actor_head
        critic_cfg = cfg.actor.model.critic_head
        self.actor_capabilities = get_prefix_actor_spec(actor_cfg.name).capabilities
        self.critic_capabilities = get_prefix_critic_spec(critic_cfg.name).capabilities
        self.num_q_heads = self.critic_capabilities.num_q_heads(critic_cfg)
        self.validate_configuration()

    def validate_configuration(self) -> None:
        """Reject actor, critic, and algorithm combinations that cannot run."""
        requirements = self.requirements
        if self.actor_capabilities.distribution != requirements.actor_distribution:
            raise ValueError(
                f"algorithm.name={self.cfg.algorithm.name!r} requires a "
                f"{requirements.actor_distribution} actor, but "
                f"actor_head.name={self.cfg.actor.model.actor_head.name!r} is "
                f"{self.actor_capabilities.distribution}."
            )
        if (
            requirements.requires_action_noise
            and not self.actor_capabilities.supports_action_noise
        ):
            raise ValueError(
                f"algorithm.name={self.cfg.algorithm.name!r} requires an actor "
                "that supports action noise."
            )
        if self.num_q_heads < requirements.min_q_heads:
            raise ValueError(
                f"algorithm.name={self.cfg.algorithm.name!r} requires at least "
                f"{requirements.min_q_heads} Q heads, got {self.num_q_heads}."
            )

    @staticmethod
    def _flatten_chunk(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _chunk_shape(self) -> tuple[int, int]:
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        return chunk_len, action_dim

    def _ref_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk_len, action_dim = self._chunk_shape()
        ref_chunk = self._flatten_chunk(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, action_dim
        )
        return ref_chunk[:, :chunk_len].reshape(ref_chunk.shape[0], -1)

    @staticmethod
    def _require_twin_q(all_q_values: torch.Tensor) -> None:
        if all_q_values.shape[-1] < 2:
            raise ValueError(
                "Prefix off-policy training requires at least two Q heads, "
                f"got Q shape {tuple(all_q_values.shape)}."
            )

    def _min_twin_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        self._require_twin_q(all_q_values)
        return torch.minimum(all_q_values[..., 0:1], all_q_values[..., 1:2])

    def _q1(self, all_q_values: torch.Tensor) -> torch.Tensor:
        self._require_twin_q(all_q_values)
        return all_q_values[..., 0:1]

    def _discounted_chunk_rewards(
        self,
        rewards: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        rewards = rewards.reshape(rewards.shape[0], -1).to(dtype)
        chunk_len = rewards.shape[-1]
        discounts = torch.pow(
            torch.as_tensor(
                self.cfg.algorithm.gamma,
                device=rewards.device,
                dtype=rewards.dtype,
            ),
            torch.arange(chunk_len, device=rewards.device, dtype=rewards.dtype),
        )
        return torch.sum(rewards * discounts, dim=-1, keepdim=True)

    def _human_mask(
        self,
        intervene_flags: torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        chunk_len, action_dim = self._chunk_shape()
        if intervene_flags is None:
            return torch.zeros(
                (batch_size, chunk_len),
                dtype=torch.bool,
                device=device,
            )

        flags = self._flatten_chunk(intervene_flags).to(device=device).bool()
        if flags.shape[-1] == chunk_len:
            return flags.reshape(batch_size, chunk_len)
        return flags.reshape(batch_size, chunk_len, action_dim).any(dim=-1)

    def _bc_metrics(
        self,
        pi: torch.Tensor,
        actions: torch.Tensor,
        ref_chunk: torch.Tensor,
        intervene_flags: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chunk_len, action_dim = self._chunk_shape()
        pi_chunk = self._flatten_chunk(pi).reshape(-1, chunk_len, action_dim)
        action_chunk = self._flatten_chunk(actions).reshape(-1, chunk_len, action_dim)
        ref_chunk = self._flatten_chunk(ref_chunk).reshape(-1, chunk_len, action_dim)
        human_mask = self._human_mask(
            intervene_flags,
            batch_size=pi_chunk.shape[0],
            device=pi_chunk.device,
        )

        bc_target = torch.where(human_mask[..., None], action_chunk, ref_chunk)
        bc_error = torch.mean(torch.square(pi_chunk - bc_target), dim=-1)
        bc_loss = torch.mean(bc_error)

        policy_mask = ~human_mask
        ref_error = torch.mean(torch.square(pi_chunk - ref_chunk), dim=-1)
        human_error = torch.mean(torch.square(pi_chunk - action_chunk), dim=-1)
        bc_ref = torch.sum(ref_error * policy_mask.to(ref_error.dtype)) / torch.clamp(
            torch.sum(policy_mask.to(ref_error.dtype)), min=1.0
        )
        bc_human = torch.sum(
            human_error * human_mask.to(human_error.dtype)
        ) / torch.clamp(torch.sum(human_mask.to(human_error.dtype)), min=1.0)

        human_ratio = torch.mean(human_mask.to(torch.float32)).item()
        return bc_loss, {
            "bc_loss": bc_loss.detach().item(),
            "bc_ref_loss": bc_ref.detach().item(),
            "bc_human_loss": bc_human.detach().item(),
            "human_mask_ratio": human_ratio,
            "policy_mask_ratio": 1.0 - human_ratio,
        }

    def _actor_objective_weights(
        self,
        update_step: int,
    ) -> tuple[float, float, dict[str, float]]:
        schedule_cfg = self.cfg.algorithm.get("actor_weight_schedule", {})
        schedule_enabled = bool(schedule_cfg.get("enable", False))
        if not schedule_enabled:
            bc_weight = float(self.cfg.algorithm.get("bc_weight", 1.0))
            q_weight = float(self.cfg.algorithm.get("q_weight", 1.0))
            return (
                bc_weight,
                q_weight,
                {
                    "bc_weight": bc_weight,
                    "q_weight": q_weight,
                    "actor_weight_schedule_enabled": 0.0,
                    "actor_weight_in_warmup": 0.0,
                    "actor_weight_ramp_progress": 1.0,
                },
            )

        warmup_updates = int(schedule_cfg.get("warmup_updates", 0))
        ramp_updates = int(schedule_cfg.get("ramp_updates", 0))
        in_warmup = update_step < warmup_updates
        warmup_bc_weight = float(
            schedule_cfg.get(
                "warmup_bc_weight",
                self.cfg.algorithm.get("bc_weight", 1.0),
            )
        )
        warmup_q_weight = float(
            schedule_cfg.get(
                "warmup_q_weight",
                self.cfg.algorithm.get("q_weight", 1.0),
            )
        )
        online_bc_weight = float(
            schedule_cfg.get(
                "online_bc_weight",
                self.cfg.algorithm.get("bc_weight", 1.0),
            )
        )
        online_q_weight = float(
            schedule_cfg.get(
                "online_q_weight",
                self.cfg.algorithm.get("q_weight", 1.0),
            )
        )

        if in_warmup:
            bc_weight = warmup_bc_weight
            q_weight = warmup_q_weight
            ramp_progress = 0.0
        elif ramp_updates > 0:
            ramp_progress = min(
                1.0,
                max(
                    0.0,
                    float(update_step - warmup_updates + 1) / float(ramp_updates),
                ),
            )
            bc_weight = warmup_bc_weight + ramp_progress * (
                online_bc_weight - warmup_bc_weight
            )
            q_weight = warmup_q_weight + ramp_progress * (
                online_q_weight - warmup_q_weight
            )
        else:
            bc_weight = online_bc_weight
            q_weight = online_q_weight
            ramp_progress = 1.0

        return (
            bc_weight,
            q_weight,
            {
                "bc_weight": bc_weight,
                "q_weight": q_weight,
                "actor_weight_schedule_enabled": 1.0,
                "actor_weight_in_warmup": float(in_warmup),
                "actor_weight_ramp_progress": ramp_progress,
            },
        )

    def _uses_cross_q(self) -> bool:
        return self.critic_capabilities.supports_cross_q

    @property
    @abstractmethod
    def requirements(self) -> PrefixAlgorithmRequirements:
        """Return the policy-head capabilities required by this algorithm."""

    @abstractmethod
    def actor_forward_kwargs(self) -> dict[str, Any]:
        """Return model options for an actor update."""

    @abstractmethod
    def target_actions(
        self,
        context: PrefixOffPolicyContext,
        next_obs: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Produce the action used by the critic target."""

    @abstractmethod
    def aggregate_actor_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        """Reduce twin-Q output for the actor objective."""

    def critic_loss(
        self,
        context: PrefixOffPolicyContext,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        use_crossq = self._uses_cross_q()
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")

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
            next_actions = self.target_actions(context, next_obs)
            if not use_crossq:
                all_qf_next_target = context.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_actions,
                )
                q_next = self._min_twin_q(all_qf_next_target)
            else:
                _, all_qf_next = context.model(
                    forward_type=ForwardType.CROSSQ_Q,
                    obs=curr_obs,
                    actions=actions,
                    next_obs=next_obs,
                    next_actions=next_actions,
                )
                q_next = self._min_twin_q(all_qf_next.detach())

            reward_target = self._discounted_chunk_rewards(
                rewards,
                context.torch_dtype,
            )
            reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
            bootstrap_discount = self.cfg.algorithm.gamma**reward_horizon
            if bootstrap_type == "always":
                target_q_values = reward_target + bootstrap_discount * q_next
            elif bootstrap_type == "standard":
                target_q_values = reward_target + not_done * bootstrap_discount * q_next
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            all_data_q_values = context.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
            )
        else:
            all_data_q_values, _ = context.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_actions,
            )

        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values,
            target_q_values.expand_as(all_data_q_values),
        )
        return critic_loss, {"q_data": all_data_q_values.mean().item()}

    def actor_loss(
        self,
        context: PrefixOffPolicyContext,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        use_crossq = self._uses_cross_q()
        curr_obs = batch["curr_obs"]
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        pi, log_pi, _ = context.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            **self.actor_forward_kwargs(),
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)

        if not use_crossq:
            all_qf_pi = context.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                detach_encoder=True,
            )
        else:
            all_qf_pi, _ = context.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                detach_encoder=True,
            )

        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(all_qf_pi.shape[-1])
        }
        qf_pi = self.aggregate_actor_q(all_qf_pi)
        metrics["q_pi"] = qf_pi.mean().item()

        ref_chunk = self._ref_chunk(curr_obs)
        bc_loss, bc_metrics = self._bc_metrics(
            pi=pi,
            actions=batch["actions"],
            ref_chunk=ref_chunk,
            intervene_flags=batch.get("intervene_flags"),
        )
        metrics.update(bc_metrics)

        bc_weight, q_weight, weight_metrics = self._actor_objective_weights(
            int(context.update_step)
        )
        actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss
        metrics.update(weight_metrics)
        metrics["weighted_q"] = (q_weight * qf_pi.mean()).detach().item()
        metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
        metrics["action_ref_abs_mean"] = (
            (self._flatten_chunk(pi) - self._flatten_chunk(ref_chunk))
            .abs()
            .mean()
            .detach()
            .item()
        )
        metrics["reference_dropout_prob"] = reference_dropout_prob
        return actor_loss, -log_pi.mean(), metrics
