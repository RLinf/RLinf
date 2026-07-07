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

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
    AsyncEmbodiedSACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class RLTACLossMixin:
    """RLT actor-critic losses on top of RLinf replay-buffer worker plumbing.

    Forward types follow the existing off-policy actor-critic API, while the
    RLT objective disables entropy/alpha and uses a fixed-std actor, min-Q
    critic target, Q1 actor objective, and BC regularization.
    """

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
        return self._flatten_chunk(obs["ref_chunk"])

    @staticmethod
    def _require_twin_q(all_q_values: torch.Tensor) -> None:
        if all_q_values.shape[-1] < 2:
            raise ValueError(
                "RLT Stage 2 requires at least two Q heads for twin-Q training, "
                f"got Q shape {tuple(all_q_values.shape)}."
            )

    def _min_twin_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        self._require_twin_q(all_q_values)
        return torch.minimum(all_q_values[..., 0:1], all_q_values[..., 1:2])

    def _q1(self, all_q_values: torch.Tensor) -> torch.Tensor:
        self._require_twin_q(all_q_values)
        return all_q_values[..., 0:1]

    def _aggregate_q(self, all_q_values: torch.Tensor, agg_q: str) -> torch.Tensor:
        if agg_q == "min":
            return self._min_twin_q(all_q_values)
        if agg_q == "mean":
            return torch.mean(all_q_values, dim=1, keepdim=True)
        raise NotImplementedError(f"{agg_q=} is not supported for RLT Stage 2.")

    def _discounted_chunk_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards = rewards.reshape(rewards.shape[0], -1)
        rewards = rewards.to(self.torch_dtype)
        chunk_len = rewards.shape[-1]
        discounts = torch.pow(
            torch.as_tensor(self.cfg.algorithm.gamma, device=rewards.device),
            torch.arange(chunk_len, device=rewards.device, dtype=rewards.dtype),
        )
        return torch.sum(rewards * discounts, dim=-1, keepdim=True)

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
        bc_ref_chunk = self._flatten_chunk(ref_chunk).reshape(
            ref_chunk.shape[0], -1, action_dim
        )[:, :chunk_len]

        if intervene_flags is None:
            human_mask = torch.zeros(
                pi_chunk.shape[:2], dtype=torch.bool, device=pi_chunk.device
            )
        else:
            human_mask = (
                self._flatten_chunk(intervene_flags)
                .to(device=pi_chunk.device)
                .bool()
                .reshape(pi_chunk.shape[0], chunk_len, action_dim)
                .any(dim=-1)
            )

        bc_target = torch.where(human_mask[..., None], action_chunk, bc_ref_chunk)
        bc_error = torch.mean(torch.square(pi_chunk - bc_target), dim=-1)
        bc_loss = torch.mean(bc_error)

        policy_mask = ~human_mask
        ref_error = torch.mean(torch.square(pi_chunk - bc_ref_chunk), dim=-1)
        human_error = torch.mean(torch.square(pi_chunk - action_chunk), dim=-1)
        bc_ref = torch.sum(ref_error * policy_mask.to(ref_error.dtype)) / torch.clamp(
            torch.sum(policy_mask.to(ref_error.dtype)), min=1.0
        )
        bc_human = torch.sum(
            human_error * human_mask.to(human_error.dtype)
        ) / torch.clamp(torch.sum(human_mask.to(human_error.dtype)), min=1.0)

        human_ratio = torch.mean(human_mask.to(torch.float32)).item()
        metrics = {
            "bc_loss": bc_loss.detach().item(),
            "bc_ref_loss": bc_ref.detach().item(),
            "bc_human_loss": bc_human.detach().item(),
            "human_mask_ratio": human_ratio,
            "policy_mask_ratio": 1.0 - human_ratio,
        }
        return bc_loss, metrics

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminations = batch["terminations"].to(self.torch_dtype)
        not_done = ~terminations.reshape(terminations.shape[0], -1).bool().any(
            dim=-1, keepdim=True
        )

        with torch.no_grad():
            next_actions, _, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
            )

            if not use_crossq:
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_actions,
                )
                q_next = self._min_twin_q(all_qf_next_target)
            else:
                _, all_qf_next = self.model(
                    forward_type=ForwardType.CROSSQ_Q,
                    obs=curr_obs,
                    actions=actions,
                    next_obs=next_obs,
                    next_actions=next_actions,
                )
                q_next = self._min_twin_q(all_qf_next.detach())

            reward_target = self._discounted_chunk_rewards(rewards)
            reward_horizon = int(rewards.reshape(rewards.shape[0], -1).shape[-1])
            bootstrap_discount = self.cfg.algorithm.gamma**reward_horizon
            if bootstrap_type == "always":
                target_q_values = reward_target + bootstrap_discount * q_next
            elif bootstrap_type == "standard":
                target_q_values = reward_target + not_done * bootstrap_discount * q_next
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")

        if not use_crossq:
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
            )
        else:
            all_data_q_values, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_actions,
            )

        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {"q_data": all_data_q_values.mean().item()}

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"

        curr_obs = batch["curr_obs"]
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        pi, log_pi, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            apply_reference_dropout=True,
            reference_dropout_prob=reference_dropout_prob,
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)

        if not use_crossq:
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                detach_encoder=True,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                detach_encoder=True,
            )

        num_q_values = all_qf_pi.shape[-1]
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(num_q_values)
        }
        qf_pi = self._q1(all_qf_pi)
        metrics["q_pi"] = qf_pi.mean().item()

        ref_chunk = self._ref_chunk(curr_obs)
        bc_loss, rlt_metrics = self._bc_metrics(
            pi=pi,
            actions=batch["actions"],
            ref_chunk=ref_chunk,
            intervene_flags=batch.get("intervene_flags", None),
        )
        metrics.update(rlt_metrics)

        entropy = -log_pi.mean()
        q_weight = float(self.cfg.algorithm.get("q_weight", 1.0))
        bc_weight = float(self.cfg.algorithm.get("bc_weight", 1.0))
        actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss
        metrics["weighted_q"] = (q_weight * qf_pi.mean()).detach().item()
        metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
        metrics["q_weight"] = q_weight
        metrics["bc_weight"] = bc_weight
        metrics["reference_dropout_prob"] = reference_dropout_prob

        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        del batch
        raise NotImplementedError(
            "RLT AC disables entropy/alpha training. Use "
            "algorithm.entropy_tuning.alpha_type=fixed_alpha."
        )


class RLTACFSDPPolicy(RLTACLossMixin, EmbodiedSACFSDPPolicy):
    pass


class AsyncRLTACFSDPPolicy(RLTACLossMixin, AsyncEmbodiedSACFSDPPolicy):
    pass
