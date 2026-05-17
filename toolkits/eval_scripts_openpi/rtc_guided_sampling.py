import math
from dataclasses import dataclass

import jax
import numpy as np
import torch

from openpi.models import model as _model
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks


@dataclass
class RTCConfig:
    num_steps: int = 5
    min_exec_horizon: int = 2
    guidance_clip: float = 5.0
    delay_buffer_size: int = 10


def build_soft_target_and_mask(
    prev_remaining: np.ndarray,
    H: int,
    action_dim: int,
    delay_steps: int,
    device: torch.device,
):
    """Build the RTC target Y and soft mask W from the previous chunk suffix."""
    target = torch.zeros((1, H, action_dim), dtype=torch.float32, device=device)
    mask = torch.zeros((1, H, 1), dtype=torch.float32, device=device)

    if prev_remaining is None or len(prev_remaining) == 0:
        return target, mask

    prev_remaining = np.asarray(prev_remaining, dtype=np.float32)
    overlap_end = min(len(prev_remaining), H)
    if overlap_end <= 0:
        return target, mask

    target[0, :overlap_end] = torch.from_numpy(prev_remaining[:overlap_end]).to(device)

    hard_end = min(max(int(delay_steps), 0), overlap_end)
    if hard_end > 0:
        mask[0, :hard_end, 0] = 1.0

    if hard_end < overlap_end:
        i = torch.arange(hard_end, overlap_end, dtype=torch.float32, device=device)
        c_i = (overlap_end - i) / float(overlap_end - hard_end + 1)
        soft_weights = c_i * (torch.expm1(c_i) / (math.e - 1.0))
        mask[0, hard_end:overlap_end, 0] = soft_weights

    return target, mask


def prepare_observation_from_policy(policy, obs_dict: dict):
    """Reuse the policy input transform and convert the result into Observation."""
    inputs = jax.tree.map(lambda x: x, obs_dict)
    inputs = policy._input_transform(inputs)
    inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(policy._pytorch_device)[None, ...],
        inputs,
    )
    observation = _model.Observation.from_dict(inputs)
    return inputs, observation


def postprocess_actions_from_policy(policy, inputs, actions_torch):
    """Reuse the policy output transform to map model actions back to env actions."""
    outputs = {
        "state": np.asarray(inputs["state"][0].detach().cpu()),
        "actions": np.asarray(actions_torch[0].detach().cpu()),
    }
    outputs = policy._output_transform(outputs)
    return outputs


def guided_sample_actions(
    policy,
    observation,
    prev_remaining: np.ndarray | None,
    delay_steps: int,
    rtc_cfg: RTCConfig,
    noise: torch.Tensor | None = None,
):
    """Run RTC-guided sampling with a cheap Jacobian-identity approximation.

    The previous implementation computed the exact vector-Jacobian product

        g = J^T c,  J = dA1_hat / dA_tau

    with ``torch.autograd.grad`` at every denoising step. That is accurate but
    expensive. This version uses the approximation ``J ~= I``, which gives

        g ~= c = (target - A1_hat) * mask

    and keeps the full denoising loop inside ``torch.no_grad()``.
    """
    model = policy._model
    device = torch.device(policy._pytorch_device)

    H = model.config.action_horizon
    action_dim = model.config.action_dim
    batch_size = observation.state.shape[0]

    if noise is None:
        noise = model.sample_noise((batch_size, H, action_dim), device)

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation,
        train=False,
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)

    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (
        "eager"
    )
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    target, mask = build_soft_target_and_mask(
        prev_remaining=prev_remaining,
        H=H,
        action_dim=action_dim,
        delay_steps=delay_steps,
        device=device,
    )

    dt = torch.tensor(1.0 / rtc_cfg.num_steps, dtype=torch.float32, device=device)
    action_sample = noise
    paper_tau = torch.tensor(0.0, dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(rtc_cfg.num_steps):
            expanded_paper_tau = paper_tau.expand(batch_size)
            model_time = 1.0 - expanded_paper_tau
            use_rtc = (
                prev_remaining is not None and len(prev_remaining) > 0 and delay_steps > 0
            )

            v_model = model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                action_sample,
                model_time,
            )
            v_paper = -v_model

            if use_rtc:
                a1_hat = action_sample + (1.0 - paper_tau) * v_paper

                # Approximate J^T c with c by using J ~= I.
                guidance_term = (target - a1_hat) * mask

                r_tau_sq = ((1.0 - paper_tau) ** 2) / (
                    paper_tau**2 + (1.0 - paper_tau) ** 2 + 1e-8
                )
                guidance_weight = (1.0 - paper_tau) / (
                    paper_tau * r_tau_sq + 1e-8
                )
                guidance_weight = torch.clamp(
                    guidance_weight,
                    max=rtc_cfg.guidance_clip,
                )
                v_paper = v_paper + guidance_weight * guidance_term

            action_sample = action_sample + dt * v_paper
            paper_tau = paper_tau + dt

    return action_sample
