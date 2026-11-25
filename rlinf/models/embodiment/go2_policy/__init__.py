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

"""Go2 actor and critic retaining RSL-RL parameter names."""

from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class Go2Policy(nn.Module, BasePolicy):
    """Go2 actor and critic with ELU activations for RLinf PPO."""

    def __init__(
        self,
        initial_checkpoint: str | None = None,
        obs_dim: int = 235,
        critic_obs_dim: int = 235,
        distribution_safety_patch: bool = False,
    ) -> None:
        super().__init__()
        self.actor = self._network(obs_dim, 12)
        self.critic = self._network(critic_obs_dim, 1)
        self.std = nn.Parameter(torch.ones(12))
        self.distribution_safety_patch = distribution_safety_patch
        if initial_checkpoint:
            state = torch.load(
                initial_checkpoint, map_location="cpu", weights_only=True
            )
            self.load_state_dict(state.get("model_state_dict", state), strict=True)

    @staticmethod
    def _network(inputs: int, outputs: int) -> nn.Sequential:
        sizes = [inputs, 512, 256, 128, outputs]
        layers = []
        for i in range(4):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < 3:
                layers.append(nn.ELU())
        return nn.Sequential(*layers)

    def preprocess_env_obs(self, env_obs: dict) -> dict:
        """Pass the same observation vector to the actor and critic."""
        device = next(self.parameters()).device
        states = env_obs["states"].to(device=device, dtype=torch.float32)
        return {"states": states, "critic_states": states}

    def forward(
        self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs: Any
    ) -> dict:
        """Dispatch the trainable forward through RLinf's policy interface."""
        return BasePolicy.forward(self, forward_type=forward_type, **kwargs)

    def _distribution(self, states: torch.Tensor) -> Normal:
        mean = self.actor(states)
        if self.distribution_safety_patch:
            mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            # Project the underlying storage to preserve FSDP's autograd views.
            self.std.data.nan_to_num_(nan=1.0, posinf=5.0, neginf=1.0e-6)
            self.std.data.clamp_(min=1.0e-6, max=5.0)
        return Normal(mean, self.std)

    def default_forward(
        self,
        forward_inputs: dict,
        compute_logprobs: bool = True,
        compute_entropy: bool = True,
        compute_values: bool = True,
        **kwargs: Any,
    ) -> dict:
        """Recompute per-joint log probabilities and the value estimate."""
        distribution = self._distribution(forward_inputs["states"])
        result = {}
        if compute_logprobs:
            result["logprobs"] = distribution.log_prob(forward_inputs["action"])
        if compute_entropy:
            result["entropy"] = distribution.entropy()
        if compute_values:
            result["values"] = self.critic(forward_inputs["critic_states"])
        if "old_mean" in forward_inputs:
            old_mean, old_std = forward_inputs["old_mean"], forward_inputs["old_std"]
            with torch.no_grad():
                result["kl_divergence"] = torch.distributions.kl_divergence(
                    Normal(old_mean, old_std), distribution
                ).sum(-1)
        return result

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs: dict,
        mode: str = "train",
        do_sample: bool | None = None,
        calculate_logprobs: bool = True,
        calculate_values: bool = True,
        return_obs: bool = True,
        **kwargs: Any,
    ) -> tuple:
        """Return actions; explicit do_sample overrides train/eval sampling."""
        obs = self.preprocess_env_obs(env_obs)
        distribution = self._distribution(obs["states"])
        mean = distribution.loc
        if mode not in ("train", "eval"):
            raise ValueError(f"Expected train or eval mode, got {mode!r}.")
        sample = mode == "train" if do_sample is None else do_sample
        action = distribution.sample() if sample else mean
        forward_inputs = {
            **obs,
            "action": action,
            "model_action": action,
            "old_mean": mean.clone(),
            "old_std": distribution.scale.clone(),
        }
        result = {
            "prev_logprobs": distribution.log_prob(action),
            "prev_values": self.critic(obs["critic_states"])
            if calculate_values
            else torch.zeros_like(action[:, :1]),
            "forward_inputs": forward_inputs,
        }
        return action[:, None, :], result
