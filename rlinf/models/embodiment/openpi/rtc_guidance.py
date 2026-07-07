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

from dataclasses import dataclass

import torch
import math


@dataclass
class RTCGuidanceContext:
    """RTC context passed from rollout runtime to the OpenPI sampler."""

    prev_model_actions: torch.Tensor | None = None  # [B, H, A_model]
    executed_horizon: int = 0
    delay_steps: int = 0

    def get_prev_remaining(self) -> torch.Tensor | None:
        if self.prev_model_actions is None:
            return None
        executed_horizon = int(max(self.executed_horizon, 0))
        if executed_horizon >= self.prev_model_actions.shape[1]:
            return None
        return self.prev_model_actions[:, executed_horizon:, :]


def build_rtc_target_and_mask(
    prev_remaining: torch.Tensor | None,
    horizon: int,
    action_dim: int,
    delay_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build overlap target and soft mask in model action space."""
    batch_size = 1 if prev_remaining is None else prev_remaining.shape[0]
    target = torch.zeros((batch_size, horizon, action_dim), device=device, dtype=dtype)
    mask = torch.zeros((batch_size, horizon, 1), device=device, dtype=dtype)

    if prev_remaining is None or prev_remaining.numel() == 0:
        return target, mask

    overlap = min(prev_remaining.shape[1], horizon)
    if overlap <= 0:
        return target, mask

    target[:, :overlap] = prev_remaining[:, :overlap].to(device=device, dtype=dtype)
    hard_end = min(max(int(delay_steps), 0), overlap)
    if hard_end > 0:
        # Actions that are likely to be executed before the new chunk arrives
        # are treated as hard constraints.
        mask[:, :hard_end, 0] = 1.0
    if hard_end < overlap:
        # Later overlap positions are softly guided so the new plan can still
        # adapt to the latest camera observation.
        i = torch.arange(hard_end, overlap, device=device, dtype=dtype)
        denom = max(float(overlap - hard_end + 1), 1.0)
        c_i = (overlap - i) / denom
        soft = c_i * (torch.expm1(c_i) / (math.e - 1.0))
        mask[:, hard_end:overlap, 0] = soft
    return target, mask


@torch.no_grad()
def sample_actions_with_rtc_guidance(
    model,
    observation,
    rtc_context: RTCGuidanceContext,
    noise: torch.Tensor | None = None,
    mode: str = "eval",
    compute_values: bool = True,
) -> dict[str, torch.Tensor]:
    """Real-world RTC sampler for OpenPI."""

    del mode

    bsize = observation.state.shape[0]
    device = observation.state.device
    num_steps = model.config.num_steps

    if noise is None:
        noise = model.sample_noise(
            (bsize, model.config.action_horizon, model.config.action_dim), device
        )

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation, train=False
    )
    prefix_output, prefix_pad_masks, past_key_values = model._build_prefix_cache(
        images, img_masks, lang_tokens, lang_masks
    )

    prev_remaining = (
        rtc_context.get_prev_remaining() if rtc_context is not None else None
    )
    target, mask = build_rtc_target_and_mask(
        prev_remaining=prev_remaining,
        horizon=model.config.action_horizon,
        action_dim=model.config.action_dim,
        delay_steps=0 if rtc_context is None else rtc_context.delay_steps,
        device=device,
        dtype=noise.dtype,
    )

    x_t = noise
    chains = [x_t]
    log_probs = []
    values = []
    denoise_inds = torch.full(
        (bsize, num_steps),
        -1,
        device=device,
        dtype=torch.long,
    )

    if model.use_vlm_value:
        values_vlm = model.get_value_from_vlm(prefix_output)

    paper_tau = torch.tensor(0.0, device=device, dtype=noise.dtype)
    dt = torch.tensor(1.0 / num_steps, device=device, dtype=noise.dtype)

    for idx in range(num_steps):
        x_t_mean, x_t_std, value_t, _ = model.sample_mean_var_val(
            x_t=x_t,
            idx=idx,
            state=state,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            sample_method="flow_ode",
            denoise_steps=num_steps,
            compute_values=compute_values,
        )

        if prev_remaining is not None and prev_remaining.numel() > 0:
            # Apply the RTC correction in denoising space before sampling the
            # next iterate. The clip keeps the guidance bounded near tau=0.
            guidance_term = (target - x_t_mean) * mask
            guidance_scale = torch.clamp(
                (1.0 - paper_tau) / torch.clamp(paper_tau + 1e-4, min=1e-4),
                max=float(model.config.rtc_guidance_clip),
            )
            x_t_mean = x_t_mean + guidance_scale * guidance_term

        noise_step = model.sample_noise(x_t.shape, device)
        x_t = x_t_mean + noise_step * x_t_std
        chains.append(x_t)
        log_probs.append(model.get_logprob_norm(x_t, x_t_mean, x_t_std))
        if value_t is not None:
            values.append(value_t)
        paper_tau = paper_tau + dt

    x_0 = x_t
    chains = torch.stack(chains, dim=1)

    if log_probs:
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : model.config.action_chunk, : model.config.action_env_dim
        ].mean(dim=1)
    else:
        log_probs = torch.zeros(
            (bsize, model.config.action_chunk, model.config.action_env_dim),
            device=device,
            dtype=noise.dtype,
        )

    if model.use_vlm_value:
        values_out = values_vlm[:, None]
    elif values:
        values_out = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)
    else:
        values_out = torch.zeros((bsize, 1), device=device, dtype=noise.dtype)

    return {
        "actions": x_0,
        "chains": chains,
        "prev_logprobs": log_probs,
        "prev_values": values_out,
        "denoise_inds": denoise_inds,
    }
