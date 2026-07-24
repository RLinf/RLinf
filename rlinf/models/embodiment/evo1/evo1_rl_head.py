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

"""RL (SDE-replay) primitives for the Evo-1 flow-matching action head.

Evo-1's action head samples with a *deterministic* Euler ODE, which has no
tractable log-prob. Following RLinf's existing flow-matching RL pattern, we turn the
denoising ODE into an SDE during rollout: at (at least) one denoising step we
inject known Gaussian noise, making that transition a Gaussian whose log-prob
we can evaluate. The full denoising trajectory (``chains``) and the stochastic
step index (``denoise_inds``) are stored so training can *replay* the same
transition under the current policy weights and recompute the log-prob.

All functions here operate directly on the public submodules of Evo-1's
``FlowmatchingActionHead`` (``time_pos_enc`` / ``_project_actions`` /
``transformer_blocks`` / ``norm_out`` / ``seq_pool_proj`` / ``mlp_head`` /
``state_encoder``) so the external Evo-1 repo does not need to be modified.

Velocity/time convention: ``x0`` is noise (t=0), ``x1`` is
data (t=1), and the head predicts a velocity ``v_t`` such that the Euler step is
``x_{t+1} = x_t + v_t * dt``.
"""

from __future__ import annotations

import random
from typing import Any, Literal, Optional

import torch


# ---------------------------------------------------------------------------- #
# Config access helper (works for dict, OmegaConf DictConfig, or namespace)
# ---------------------------------------------------------------------------- #
def cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ---------------------------------------------------------------------------- #
# Gaussian log-prob for one SDE transition step
# ---------------------------------------------------------------------------- #
def get_logprob_norm(
    sample: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    safe: bool = False,
) -> torch.Tensor:
    """Element-wise Gaussian log-prob. Dims with ``sigma == 0`` contribute 0."""
    if safe:
        from torch.distributions import Normal

        return Normal(loc=mu, scale=sigma).log_prob(sample)
    mask = sigma == 0
    sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
    constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
        2 * torch.pi * torch.ones_like(sample)
    )
    exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
    log_prob = constant_term + exponent_term
    log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
    return log_prob


# ---------------------------------------------------------------------------- #
# Prompt <-> byte-tensor (lossless, keeps forward_inputs all-tensor)
# ---------------------------------------------------------------------------- #
def bytes_from_prompts(
    prompts: list[str], max_len: int = 512, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a list of prompt strings into a fixed-width UTF-8 byte tensor.

    Returns ``(prompt_bytes[B, max_len] long, prompt_len[B] long)``. This keeps
    the rollout ``forward_inputs`` dict all-tensor (RLinf's rollout->train merge
    pipeline rejects strings / variable-length lists).
    """
    bsz = len(prompts)
    buf = torch.zeros(bsz, max_len, dtype=torch.long)
    lens = torch.zeros(bsz, dtype=torch.long)
    for i, p in enumerate(prompts):
        raw = (p or "").encode("utf-8")[:max_len]
        n = len(raw)
        lens[i] = n
        if n:
            buf[i, :n] = torch.tensor(list(raw), dtype=torch.long)
    if device is not None:
        buf = buf.to(device)
        lens = lens.to(device)
    return buf, lens


def prompts_from_bytes(
    prompt_bytes: torch.Tensor, prompt_len: torch.Tensor
) -> list[str]:
    """Losslessly reconstruct prompt strings from the byte tensor."""
    out: list[str] = []
    bsz = prompt_bytes.shape[0]
    for i in range(bsz):
        n = int(prompt_len[i].item())
        raw = bytes(prompt_bytes[i, :n].detach().cpu().tolist())
        out.append(raw.decode("utf-8", errors="ignore"))
    return out


# ---------------------------------------------------------------------------- #
# Evo-1 head single-step velocity (extracted from FlowmatchingActionHead)
# ---------------------------------------------------------------------------- #
def build_context_tokens(
    action_head,
    fused_tokens: torch.Tensor,
    state: Optional[torch.Tensor],
    embodiment_id: torch.Tensor,
) -> torch.Tensor:
    """Concatenate state embedding onto fused VLM tokens (once per forward).

    Mirrors ``FlowmatchingActionHead.get_action`` context construction.
    """
    context = fused_tokens
    if state is not None and action_head.state_encoder is not None:
        state_emb = action_head.state_encoder(state, embodiment_id).unsqueeze(1)
        context = torch.cat([context, state_emb], dim=1)
    return context


def flow_velocity(
    action_head,
    context_tokens: torch.Tensor,
    x_t: torch.Tensor,
    idx,
    denoise_steps: int,
    embodiment_id: torch.Tensor,
) -> torch.Tensor:
    """Predict the flow velocity ``v_t`` at denoise step ``idx``.

    This is the body of ``FlowmatchingActionHead.get_action``'s loop, lifted out
    as a pure function. ``x_t`` is ``[B, H, per_dim]``; ``idx`` is an int (rollout)
    or a LongTensor ``[B]`` (replay). Returns ``v_t`` shaped ``[B, H, per_dim]``.
    """
    bsz = context_tokens.shape[0]
    device = context_tokens.device
    horizon = action_head.horizon
    per_dim = action_head.per_action_dim
    target_dtype = action_head.dtype

    # Continuous time -> sinusoidal time embedding, matching get_action:
    #   t = idx / N ; time_index = min(int(t*999), 999).
    pe = action_head.time_pos_enc(1000).to(device).squeeze(0)  # [1000, embed]
    if isinstance(idx, int):
        t_cont = idx / float(denoise_steps)
        time_index = min(int(t_cont * 999), 999)
        time_emb = pe[time_index].to(dtype=target_dtype)
        time_emb = time_emb.unsqueeze(0).repeat(bsz, 1)  # [B, embed]
    else:
        t_cont = idx.to(device).float() / float(denoise_steps)  # [B]
        time_index = (t_cont * 999).long().clamp_(0, 999)  # [B]
        time_emb = pe[time_index].to(dtype=target_dtype)  # [B, embed]

    action_tokens = action_head._project_actions(x_t, embodiment_id)
    action_tokens = action_tokens.to(dtype=target_dtype)
    context_tokens = context_tokens.to(dtype=target_dtype)

    x = action_tokens
    for block in action_head.transformer_blocks:
        x = block(x, context_tokens, time_emb)
    x = action_head.norm_out(x)

    if horizon > 1:
        x_flat = x.reshape(bsz, -1)
        x_pooled = action_head.seq_pool_proj(x_flat)
    else:
        x_pooled = x.squeeze(1)

    pred = action_head.mlp_head(x_pooled, embodiment_id)  # [B, H*per_dim]
    return pred.view(bsz, horizon, per_dim)


# ---------------------------------------------------------------------------- #
# SDE mean/std (flow_sde weights, applied to Evo-1's v_t)
# ---------------------------------------------------------------------------- #
def flow_sde_mean_std(
    x_t: torch.Tensor,
    v_t: torch.Tensor,
    idx,
    denoise_steps: int,
    noise_level: float,
    mode: Literal["train", "eval"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the Gaussian transition ``(mean, std)`` for one denoising step.

    ``eval`` reproduces the deterministic Euler step exactly
    (``mean = x_t + v_t * dt``, ``std = 0``). ``train`` uses the ``flow_sde``
    weights (the standard flow-matching SDE mean/variance formulas).
    """
    device = x_t.device
    dtype = x_t.dtype
    num = denoise_steps
    timesteps = torch.linspace(0, 1, num + 1, device=device, dtype=dtype)  # [N+1]

    if isinstance(idx, int):
        idx_t = torch.tensor([idx], device=device).expand(x_t.shape[0])
    else:
        idx_t = idx.to(device)

    t_input = timesteps[idx_t][:, None, None].expand_as(x_t)
    delta = (timesteps[idx_t + 1] - timesteps[idx_t])[:, None, None].expand_as(x_t)

    # x0 = noise (t=0), x1 = data (t=1).
    x0_pred = x_t - v_t * t_input
    x1_pred = x_t + v_t * (1 - t_input)

    if mode == "eval":
        x0_weight = 1 - (t_input + delta)
        x1_weight = t_input + delta
        x_t_std = torch.zeros_like(t_input)
    else:  # flow_sde
        noise_level_t = torch.as_tensor(noise_level, device=device, dtype=dtype)
        sigmas = (
            noise_level_t
            * torch.sqrt(
                (1 - timesteps)
                / torch.where(timesteps == 0, timesteps[1], timesteps)
            )[:-1]
        )  # [N]
        sigma_i = sigmas[idx_t][:, None, None].expand_as(x_t)
        x0_weight = (
            torch.ones_like(t_input)
            - (t_input + delta)
            - sigma_i**2 * delta / (2 * (1 - t_input))
        )
        x1_weight = t_input + delta
        x_t_std = torch.sqrt(delta) * sigma_i

    x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
    return x_t_mean, x_t_std


# ---------------------------------------------------------------------------- #
# Rollout: SDE denoising trajectory
# ---------------------------------------------------------------------------- #
def rl_denoise_sample(
    action_head,
    fused_tokens: torch.Tensor,
    state: Optional[torch.Tensor],
    embodiment_id: torch.Tensor,
    action_mask_3d: torch.Tensor,
    denoise_steps: int,
    rl_cfg: Any,
    mode: Literal["train", "eval"] = "train",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the SDE denoising loop and record the trajectory.

    Returns ``(x_0, chains, denoise_inds, log_probs)`` where
      * ``x_0``          -- final normalized action ``[B, H, per_dim]``
      * ``chains``       -- ``[B, N+1, H, per_dim]`` full trajectory
      * ``denoise_inds`` -- ``[B, N]`` the stochastic step index (same value per row)
      * ``log_probs``    -- ``[B, N (+1 if joint), H, per_dim]`` per-step log-probs
    """
    device = fused_tokens.device
    bsz = fused_tokens.shape[0]
    horizon = action_head.horizon
    per_dim = action_head.per_action_dim
    dtype = action_head.dtype

    joint = bool(cfg_get(rl_cfg, "joint_logprob", False))
    noise_level = float(cfg_get(rl_cfg, "noise_level", 0.1))
    ignore_last = bool(cfg_get(rl_cfg, "ignore_last", True))

    context_tokens = build_context_tokens(
        action_head, fused_tokens, state, embodiment_id
    )
    action_mask_3d = action_mask_3d.to(device=device, dtype=dtype)

    # Initial x_t: uniform [-1, 1], matching Evo-1's training/inference noise.
    x_t = torch.rand(bsz, horizon, per_dim, device=device, dtype=dtype) * 2 - 1
    x_t = x_t * action_mask_3d

    num = denoise_steps
    chains = [x_t]
    log_probs = []

    if joint:
        log_probs.append(
            get_logprob_norm(x_t, torch.zeros_like(x_t), torch.ones_like(x_t))
        )

    # Choose which step(s) are stochastic. Non-joint: one random step, shared
    # across the batch and repeated across the step axis.
    if mode == "train":
        if joint:
            denoise_inds = torch.arange(num)
        else:
            hi = num - 2 if ignore_last else num - 1
            hi = max(hi, 0)
            denoise_inds = torch.tensor([random.randint(0, hi)] * num)
    else:
        denoise_inds = torch.tensor([-1] * num)
    denoise_inds = denoise_inds[None].repeat(bsz, 1).to(device)

    for idx in range(num):
        step_mode = "train" if idx == int(denoise_inds[0][idx].item()) else "eval"
        v_t = flow_velocity(
            action_head, context_tokens, x_t * action_mask_3d, idx, num, embodiment_id
        )
        x_t_mean, x_t_std = flow_sde_mean_std(
            x_t, v_t, idx, num, noise_level, step_mode
        )
        x_t_mean = x_t_mean * action_mask_3d
        x_t_std = x_t_std * action_mask_3d
        eps = torch.randn(x_t.shape, device=device, dtype=dtype)
        x_t = x_t_mean + eps * x_t_std
        x_t = x_t * action_mask_3d
        log_probs.append(get_logprob_norm(x_t, x_t_mean, x_t_std))
        chains.append(x_t)

    x_0 = x_t
    chains = torch.stack(chains, dim=1)  # [B, N+1, H, per_dim]
    log_probs = torch.stack(log_probs, dim=1)  # [B, N(+1), H, per_dim]
    return x_0, chains, denoise_inds, log_probs


# ---------------------------------------------------------------------------- #
# Replay: recompute log-probs on stored chains under current weights
# ---------------------------------------------------------------------------- #
def rl_replay_logprob(
    action_head,
    fused_tokens: torch.Tensor,
    state: Optional[torch.Tensor],
    embodiment_id: torch.Tensor,
    action_mask_3d: torch.Tensor,
    chains: torch.Tensor,
    denoise_inds: torch.Tensor,
    denoise_steps: int,
    rl_cfg: Any,
) -> torch.Tensor:
    """Replay the stored ``chains`` and recompute per-step log-probs.

    Training-time counterpart of ``rl_denoise_sample`` (recomputes log-probs).
    Returns ``chains_log_probs`` shaped ``[B, num_steps (+1 if joint), H, per_dim]``.
    """
    device = fused_tokens.device
    bsz = fused_tokens.shape[0]
    dtype = action_head.dtype

    joint = bool(cfg_get(rl_cfg, "joint_logprob", False))
    noise_level = float(cfg_get(rl_cfg, "noise_level", 0.1))
    num = denoise_steps

    context_tokens = build_context_tokens(
        action_head, fused_tokens, state, embodiment_id
    )
    action_mask_3d = action_mask_3d.to(device=device, dtype=dtype)

    chains_log_probs = []
    if joint:
        num_steps = num
        chains_log_probs.append(
            get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
        )
    else:
        num_steps = 1

    arange = torch.arange(bsz, device=device)
    for idx in range(num_steps):
        denoise_ind = denoise_inds[:, idx]  # [B]
        chains_pre = chains[arange, denoise_ind]  # [B, H, per_dim]
        chains_next = chains[arange, denoise_ind + 1]
        v_t = flow_velocity(
            action_head,
            context_tokens,
            chains_pre * action_mask_3d,
            denoise_ind,
            num,
            embodiment_id,
        )
        x_t_mean, x_t_std = flow_sde_mean_std(
            chains_pre, v_t, denoise_ind, num, noise_level, mode="train"
        )
        x_t_mean = x_t_mean * action_mask_3d
        x_t_std = x_t_std * action_mask_3d
        chains_log_probs.append(get_logprob_norm(chains_next, x_t_mean, x_t_std))

    return torch.stack(chains_log_probs, dim=1)
