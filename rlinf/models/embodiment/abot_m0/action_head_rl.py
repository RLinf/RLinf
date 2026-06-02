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

"""RL action head wrapper for ABot-M0."""

import random
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.prediction_type = rl_head_config.get("prediction_type", "velocity")
        if self.prediction_type not in ("velocity", "x1"):
            raise ValueError(
                f"Invalid prediction_type {self.prediction_type!r}; "
                "expected 'velocity' or 'x1'."
            )
        self._timesteps_cache: dict[tuple[int, str, int], torch.Tensor] = {}

        if rl_head_config.get("add_value_head", False):
            self.value_head = ValueHead(
                input_dim=vl_hidden_size * 2,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

    def _get_or_build_timesteps(
        self, num_steps: int, device: torch.device
    ) -> torch.Tensor:
        device_index = -1 if device.index is None else int(device.index)
        cache_key = (num_steps, device.type, device_index)
        cached = self._timesteps_cache.get(cache_key)
        if cached is None:
            cached = torch.linspace(
                0,
                1,
                num_steps + 1,
                device=device,
            )
            self._timesteps_cache[cache_key] = cached
        return cached

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

    def _encode_state_features(
        self, state: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if state is None or self.base.state_encoder is None:
            return None
        encoder = self.base.state_encoder
        encoder_module = getattr(encoder, "module", encoder)
        encoder_module = getattr(encoder_module, "_fsdp_wrapped_module", encoder_module)

        target_device = None
        target_dtype = None
        layer1 = getattr(encoder_module, "layer1", None)
        if (
            layer1 is not None
            and hasattr(layer1, "weight")
            and layer1.weight is not None
        ):
            target_device = layer1.weight.device
            target_dtype = layer1.weight.dtype
        else:
            first_param = next(iter(encoder_module.parameters()), None)
            if first_param is not None:
                target_device = first_param.device
                target_dtype = first_param.dtype

        if target_device is not None and target_dtype is not None:
            state = state.to(device=target_device, dtype=target_dtype)
        return encoder(state)

    def _denoise_step(
        self,
        vl_embs: torch.Tensor,
        x_t: torch.Tensor,
        state_features: Optional[torch.Tensor],
        idx: int | torch.Tensor,
        num_steps: int,
        mode: Literal["train", "eval"] = "train",
    ):
        """Run one denoising step and return step mean/std."""
        device = vl_embs.device
        batch_size = vl_embs.shape[0]

        action_encoder = self.base.action_encoder
        action_encoder_param = next(iter(action_encoder.parameters()), None)
        if action_encoder_param is not None:
            x_t = x_t.to(
                device=action_encoder_param.device,
                dtype=action_encoder_param.dtype,
            )

        if isinstance(idx, int):
            idx_tensor = torch.full(
                (batch_size,),
                idx,
                device=device,
                dtype=torch.long,
            )
        else:
            idx_tensor = idx.to(device=device, dtype=torch.long).reshape(-1)
            if idx_tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Expected idx tensor with shape [{batch_size}], got {tuple(idx_tensor.shape)}"
                )

        t_cont = idx_tensor.to(dtype=x_t.dtype) / float(num_steps)
        dt = 1.0 / num_steps
        timesteps_tensor = torch.clamp(
            (t_cont * self.num_timestep_buckets).to(dtype=torch.long),
            min=0,
            max=self.num_timestep_buckets - 1,
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

        try:
            model_dtype = next(self.base.model.parameters()).dtype
        except StopIteration:
            model_buffer = next(self.base.model.buffers(), None)
            model_dtype = (
                model_buffer.dtype if model_buffer is not None else sa_embs.dtype
            )
        if sa_embs.dtype != model_dtype:
            sa_embs = sa_embs.to(dtype=model_dtype)
        if vl_embs.dtype != model_dtype:
            vl_embs = vl_embs.to(dtype=model_dtype)

        timestep_encoder = getattr(self.base.model, "timestep_encoder", None)
        if timestep_encoder is not None and hasattr(timestep_encoder, "compute_dtype"):
            timestep_encoder.compute_dtype = model_dtype

        model_output = self.base.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )
        decoded = self.base.action_decoder(model_output)
        decoded = decoded[:, -self.action_horizon :]

        t_broadcast = t_cont.view(batch_size, 1, 1).expand_as(x_t)
        dt_tensor = torch.full_like(x_t, dt)

        if self.prediction_type == "velocity":
            pred_velocity = decoded
        else:  # "x1"
            pred_velocity = decoded - x_t
            pred_velocity = pred_velocity / (1.0 - t_broadcast).clamp_min(self.t_eps)

        x0_pred = x_t - pred_velocity * t_broadcast
        x1_pred = x_t + pred_velocity * (1.0 - t_broadcast)

        if mode == "eval":
            x0_weight = 1.0 - (t_broadcast + dt_tensor)
            x1_weight = t_broadcast + dt_tensor
            x_t_mean = (x0_weight * x0_pred + x1_weight * x1_pred).to(x_t.dtype)
            x_t_std = torch.zeros_like(x_t_mean, dtype=x_t.dtype)
        else:
            if self.noise_method == "flow_sde":
                noise_level = torch.tensor(
                    self.noise_level,
                    device=device,
                    dtype=x_t.dtype,
                )
                timesteps = self._get_or_build_timesteps(num_steps, device).to(
                    dtype=x_t.dtype
                )

                sigmas = (
                    noise_level
                    * torch.sqrt(
                        (1 - timesteps)
                        / torch.where(timesteps == 0, timesteps[1], timesteps)
                    )[:-1]
                )
                sigma_i = (
                    sigmas[idx_tensor]
                    .view(batch_size, 1, 1)
                    .expand_as(x_t)
                    .to(x_t.dtype)
                )

                x0_weight = (
                    torch.ones_like(t_broadcast)
                    - (t_broadcast + dt_tensor)
                    - sigma_i**2 * dt_tensor / (2 * (1.0 - t_broadcast).clamp_min(1e-8))
                )
                x1_weight = t_broadcast + dt_tensor
                x_t_mean = (x0_weight * x0_pred + x1_weight * x1_pred).to(x_t.dtype)
                x_t_std = (torch.sqrt(dt_tensor) * sigma_i).to(x_t.dtype)
            else:
                x_t_mean = (x_t + dt_tensor * pred_velocity).to(x_t.dtype)
                x_t_std = (self.noise_level * torch.ones_like(x_t_mean)).to(x_t.dtype)

        return x_t_mean, x_t_std

    def get_value(
        self,
        vl_embs: torch.Tensor,
        state_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute value estimate from VL embeddings."""
        if not hasattr(self, "value_head"):
            return torch.zeros(
                vl_embs.shape[0], 1, device=vl_embs.device, dtype=vl_embs.dtype
            )
        pooled_vl = vl_embs.mean(dim=1)  # [B, H]
        if state_features is not None:
            state_features_value = state_features.reshape(state_features.shape[0], -1)
            state_features_value = state_features_value.to(
                device=pooled_vl.device, dtype=pooled_vl.dtype
            )
            pooled_state = F.adaptive_avg_pool1d(
                state_features_value.unsqueeze(1), pooled_vl.shape[-1]
            ).squeeze(1)
        else:
            pooled_state = torch.zeros_like(pooled_vl)
        value_input = torch.cat((pooled_vl, pooled_state), dim=1)
        value_head_param = next(iter(self.value_head.parameters()), None)
        if value_head_param is not None:
            value_input = value_input.to(
                device=value_head_param.device,
                dtype=value_head_param.dtype,
            )
        return self.value_head(value_input)  # [B, 1]

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

        state_features = self._encode_state_features(state)

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

        values = self.get_value(vl_embs, state_features)  # [B, 1]

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

        state_features = self._encode_state_features(state)

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
                idx=di,
                num_steps=self.num_inference_timesteps,
                mode="train",
            )

            lp = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(lp)

        log_probs = torch.stack(chains_log_probs, dim=1)  # [B, num_steps(+1), T, D]

        if compute_values:
            values = self.get_value(vl_embs, state_features)
        else:
            values = torch.zeros(
                batch_size,
                1,
                device=vl_embs.device,
                dtype=vl_embs.dtype,
            )

        return log_probs, values
