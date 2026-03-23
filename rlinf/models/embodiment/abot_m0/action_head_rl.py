# Copyright 2025 The RLinf Authors.
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

"""RL action head wrapper for ABot-M0."""

import random
from typing import Any, Literal, Optional

import torch
import torch.nn as nn

from rlinf.models.embodiment.modules.value_head import ValueHead


class AMLFlowMatchingActionHeadRL(nn.Module):
    """Adds rollout logprob/value utilities to ABot-M0 action head."""

    def __init__(
        self,
        base_action_head: nn.Module,
        rl_head_config: dict[str, Any],
        vl_hidden_size: int,
    ):
        super().__init__()
        self.base = base_action_head
        self.rl_config = rl_head_config

        self.action_dim = self.base.action_dim
        self.action_horizon = self.base.action_horizon
        self.num_inference_timesteps = self.base.num_inference_timesteps
        self.num_timestep_buckets = self.base.num_timestep_buckets
        self.t_eps = float(rl_head_config.get("t_eps", self.base.t_eps))
        self.base.t_eps = self.t_eps

        noise_level = rl_head_config.get("noise_level", 0.5)
        self.noise_level = noise_level
        self.noise_method = rl_head_config.get("noise_method", "flow_sde")
        self.joint_logprob = rl_head_config.get("joint_logprob", False)

        if rl_head_config.get("add_value_head", False):
            self.value_head = ValueHead(
                input_dim=vl_hidden_size,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

    def get_logprob_norm(
        self,
        sample: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability under a Normal distribution."""
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def _denoise_step(
        self,
        vl_embs: torch.Tensor,
        x_t: torch.Tensor,
        state_features: Optional[torch.Tensor],
        idx: int,
        num_steps: int,
        mode: Literal["train", "eval"] = "train",
    ):
        """Run one denoising step and return step mean/std."""
        device = vl_embs.device
        batch_size = vl_embs.shape[0]

        t_cont = idx / float(num_steps)
        dt = 1.0 / num_steps
        t_discretized = int(t_cont * self.num_timestep_buckets)
        timesteps_tensor = torch.full(
            (batch_size,),
            t_discretized,
            device=device,
            dtype=torch.long,
        )

        action_features = self.base.action_encoder(x_t, timesteps_tensor)

        if self.base.config.add_pos_embed:
            pos_ids = torch.arange(
                action_features.shape[1],
                dtype=torch.long,
                device=device,
            )
            pos_embs = self.base.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.base.future_tokens.weight.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
        if state_features is not None:
            sa_embs = torch.cat(
                (state_features, future_tokens, action_features),
                dim=1,
            )
        else:
            sa_embs = torch.cat((future_tokens, action_features), dim=1)

        model_output = self.base.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        pred = self.base.action_decoder(model_output)
        pred_actions = pred[:, -self.action_horizon :]

        t_broadcast = t_cont * torch.ones(
            1,
            1,
            1,
            device=device,
            dtype=x_t.dtype,
        )
        pred_velocity = (pred_actions - x_t) / (1.0 - t_broadcast).clamp_min(
            self.t_eps,
        )

        x_t_mean = x_t + dt * pred_velocity

        if mode == "eval":
            x_t_std = torch.zeros_like(x_t_mean)
        else:
            if self.noise_method == "flow_sde":
                noise_level = torch.tensor(
                    self.noise_level,
                    device=device,
                    dtype=x_t.dtype,
                )
                timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
                sigma_i = noise_level * torch.sqrt(
                    (1 - timesteps[idx + 1]) / timesteps[idx + 1].clamp_min(1e-8),
                )
                x_t_std = (dt**0.5) * sigma_i * torch.ones_like(x_t_mean)
            else:
                x_t_std = self.noise_level * torch.ones_like(x_t_mean)

        return x_t_mean, x_t_std

    def get_value(self, vl_embs: torch.Tensor) -> torch.Tensor:
        """Compute value estimate from VL embeddings."""
        if not hasattr(self, "value_head"):
            return torch.zeros(
                vl_embs.shape[0], 1, device=vl_embs.device, dtype=vl_embs.dtype
            )
        pooled = vl_embs.mean(dim=1)  # [B, H]
        return self.value_head(pooled)  # [B, 1]

    @torch.no_grad()
    def get_rl_action(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        mode: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run rollout denoising loop and return actions plus training cache."""
        device = vl_embs.device
        batch_size = vl_embs.shape[0]
        num_steps = self.num_inference_timesteps

        state_features = (
            self.base.state_encoder(state)
            if state is not None and self.base.state_encoder is not None
            else None
        )

        x_t = torch.randn(
            batch_size,
            self.action_horizon,
            self.action_dim,
            device=device,
            dtype=vl_embs.dtype,
        )

        chains = [x_t]
        log_probs = []

        if self.joint_logprob:
            init_lp = self.get_logprob_norm(
                x_t,
                torch.zeros_like(x_t),
                torch.ones_like(x_t),
            )
            log_probs.append(init_lp)

        if mode == "train":
            if self.joint_logprob:
                denoise_inds = torch.arange(num_steps, device=device)
            else:
                max_idx = num_steps - 1
                if self.noise_method == "flow_sde" and self.rl_config.get(
                    "ignore_last", False
                ):
                    max_idx = max(num_steps - 2, 0)
                sampled_idx = random.randint(0, max_idx)
                denoise_inds = torch.full((num_steps,), sampled_idx, device=device)
        else:
            denoise_inds = torch.full((num_steps,), -1, device=device)
        denoise_inds = denoise_inds[None].repeat(batch_size, 1)

        for idx in range(num_steps):
            step_mode = mode
            if mode == "train" and idx != denoise_inds[0, idx].item():
                step_mode = "eval"

            x_t_mean, x_t_std = self._denoise_step(
                vl_embs,
                x_t,
                state_features,
                idx,
                num_steps,
                mode=step_mode,
            )

            noise = torch.randn_like(x_t)
            x_t = x_t_mean + noise * x_t_std

            lp = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
            log_probs.append(lp)
            chains.append(x_t)

        actions = x_t
        chains = torch.stack(chains, dim=1)  # [B, num_steps+1, T, D]
        log_probs = torch.stack(log_probs, dim=1)  # [B, num_steps(+1 if joint), T, D]

        values = self.get_value(vl_embs)  # [B, 1]

        return actions, {
            "actions": actions,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def forward(
        self,
        vl_embs: torch.Tensor,
        state: Optional[torch.Tensor],
        chains: torch.Tensor,
        denoise_inds: torch.Tensor,
        compute_values: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recompute rollout logprobs from cached denoising chains."""
        batch_size = vl_embs.shape[0]

        state_features = (
            self.base.state_encoder(state)
            if state is not None and self.base.state_encoder is not None
            else None
        )

        chains_log_probs = []

        if self.joint_logprob:
            num_steps = self.num_inference_timesteps
            init_lp = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            chains_log_probs.append(init_lp)
        else:
            num_steps = 1

        for s in range(num_steps):
            di = denoise_inds[:, s]
            chains_pre = chains[torch.arange(batch_size), di]
            chains_next = chains[torch.arange(batch_size), di + 1]

            x_t_mean, x_t_std = self._denoise_step(
                vl_embs,
                chains_pre,
                state_features,
                idx=di[0].item(),
                num_steps=self.num_inference_timesteps,
                mode="train",
            )

            lp = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(lp)

        log_probs = torch.stack(chains_log_probs, dim=1)  # [B, num_steps(+1), T, D]

        if compute_values:
            values = self.get_value(vl_embs)
        else:
            values = torch.zeros(
                batch_size,
                1,
                device=vl_embs.device,
                dtype=vl_embs.dtype,
            )

        return log_probs, values
