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

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.prefix.heads.registry import (
    build_prefix_actor,
    build_prefix_critic,
)


class PrefixPolicy(nn.Module, BasePolicy):
    """Compose registered actor and critic heads over frozen prefix features."""

    def __init__(
        self,
        *,
        prefix_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None,
        actor_head_cfg: Any,
        critic_head_cfg: Any,
    ) -> None:
        super().__init__()
        self.prefix_dim = int(prefix_dim)
        self.proprio_dim = int(proprio_dim)
        self.step_action_dim = int(action_dim)
        self.chunk_len = int(num_action_chunks)
        self.ref_chunk_len = (
            self.chunk_len
            if ref_num_action_chunks is None
            else int(ref_num_action_chunks)
        )
        if self.ref_chunk_len < self.chunk_len:
            raise ValueError(
                "ref_num_action_chunks must be >= num_action_chunks, got "
                f"{self.ref_chunk_len} < {self.chunk_len}."
            )

        self.action_dim = self.step_action_dim
        self.num_action_chunks = self.chunk_len
        self.flat_action_dim = self.chunk_len * self.step_action_dim
        self.state_dim = self.prefix_dim + self.proprio_dim
        self.torch_compile_enabled = False

        actor_name = actor_head_cfg.get("name", None)
        critic_name = critic_head_cfg.get("name", None)
        if actor_name is None or critic_name is None:
            raise ValueError(
                "prefix_policy requires actor_head.name and critic_head.name."
            )
        self.actor_head = build_prefix_actor(
            actor_name,
            state_dim=self.state_dim,
            reference_dim=self.flat_action_dim,
            action_dim=self.flat_action_dim,
            cfg=actor_head_cfg,
        )
        # SAC optimizer filtering and target updates identify critic parameters
        # through the stable ``q_head`` path.
        self.q_head = build_prefix_critic(
            critic_name,
            state_dim=self.state_dim,
            action_dim=self.flat_action_dim,
            cfg=critic_head_cfg,
        )

    def preprocess_env_obs(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Move tensor observations to the policy device."""
        device = next(self.parameters()).device
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in env_obs.items()
        }

    @staticmethod
    def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _state(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [
                self._flatten_batch(obs["z_rl"]),
                self._flatten_batch(obs["proprio"]),
            ],
            dim=-1,
        )

    def _reference(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        reference = self._flatten_batch(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0],
            -1,
            self.step_action_dim,
        )
        reference = reference[:, : self.chunk_len]
        return reference.reshape(reference.shape[0], -1)

    def _format_chunk_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(-1, self.chunk_len, self.step_action_dim)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        """Dispatch the forward pass required by the off-policy worker."""
        obs = kwargs.get("obs")
        if obs is not None:
            kwargs["obs"] = self.preprocess_env_obs(obs)
        next_obs = kwargs.get("next_obs")
        if next_obs is not None:
            kwargs["next_obs"] = self.preprocess_env_obs(next_obs)

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.CROSSQ:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.CROSSQ_Q:
            return self.crossq_q_forward(**kwargs)
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Unsupported forward_type: {forward_type}")

    def default_forward(self, **kwargs):
        """Reject PPO-style forwards, which this policy does not implement."""
        del kwargs
        raise NotImplementedError(
            "PrefixPolicy does not use PPO-style default_forward."
        )

    def _actor_forward_from_processed_tensors(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        *,
        deterministic: bool,
        apply_reference_dropout: bool,
        reference_dropout_prob: float | None,
        apply_action_noise: bool | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor_head(
            state,
            reference,
            deterministic=deterministic,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
            apply_action_noise=apply_action_noise,
        )

    def sac_forward(
        self,
        obs: dict[str, torch.Tensor],
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float | None = None,
        deterministic: bool = False,
        apply_action_noise: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Produce actions and log probabilities for off-policy training."""
        del kwargs
        actions, logprobs = self._actor_forward_from_processed_tensors(
            self._state(obs),
            self._reference(obs),
            deterministic=deterministic,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
            apply_action_noise=apply_action_noise,
        )
        return actions, logprobs, None

    def sac_q_forward(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        shared_feature=None,
        detach_encoder: bool = False,
    ) -> torch.Tensor:
        """Evaluate the configured critic on one state-action batch."""
        del shared_feature
        state = self._state(obs)
        if detach_encoder:
            state = state.detach()
        output = self.q_head(state, self._flatten_batch(actions))
        if isinstance(output, tuple):
            return output[0]
        return output

    def crossq_q_forward(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
        next_obs: dict[str, torch.Tensor] | None = None,
        next_actions: torch.Tensor | None = None,
        shared_feature=None,
        detach_encoder: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate current and next-state Q values through a CrossQ head."""
        del shared_feature
        state = self._state(obs)
        next_state = self._state(next_obs) if next_obs is not None else None
        if detach_encoder:
            state = state.detach()
            if next_state is not None:
                next_state = next_state.detach()
        output = self.q_head(
            state,
            self._flatten_batch(actions),
            next_state=next_state,
            next_actions=(
                self._flatten_batch(next_actions) if next_actions is not None else None
            ),
        )
        if isinstance(output, tuple):
            return output
        if next_state is None or next_actions is None:
            return output, output.new_zeros(output.shape)
        next_q = self.q_head(next_state, self._flatten_batch(next_actions))
        if isinstance(next_q, tuple):
            next_q = next_q[0]
        return output, next_q

    def sft_forward(self, data: dict[str, Any], **kwargs) -> torch.Tensor:
        """Return per-action behavior-cloning error."""
        del kwargs
        obs = data["obs"] if "obs" in data else data
        target_actions = self._flatten_batch(
            data["action"] if "action" in data else data["actions"]
        )
        predicted_actions = self.actor_head.mean_action(
            self._state(obs),
            self._reference(obs),
        )
        return F.mse_loss(predicted_actions, target_actions, reduction="none")

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        """Predict an action chunk from a compact prefix observation."""
        del calculate_logprobs, calculate_values, kwargs
        obs = self.preprocess_env_obs(env_obs)
        actions, logprobs, _ = self.sac_forward(
            obs,
            deterministic=(mode == "eval"),
            apply_action_noise=(mode != "eval"),
        )
        chunk_actions = self._format_chunk_actions(actions)
        forward_inputs = {"action": actions, "model_action": actions}
        if return_obs:
            forward_inputs.update(obs)
        return chunk_actions, {
            "prev_logprobs": logprobs,
            "prev_values": torch.zeros_like(logprobs[..., :1]),
            "forward_inputs": forward_inputs,
        }

    def enable_torch_compile(
        self,
        mode: str = "max-autotune-no-cudagraphs",
    ) -> None:
        """Compile the actor-only tensor path once."""
        if self.torch_compile_enabled:
            return
        self._actor_forward_from_processed_tensors = torch.compile(
            self._actor_forward_from_processed_tensors,
            mode=mode,
        )
        self.torch_compile_enabled = True

    def set_critic_requires_grad(self, requires_grad: bool) -> None:
        """Set critic parameter trainability."""
        for parameter in self.q_head.parameters():
            parameter.requires_grad_(requires_grad)
