# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""RLinf RTC guidance for the frozen Stage1 OpenPI Euler sampler.

The overlap target and mask are RLinf's own ``build_rtc_target_and_mask``:
hard for the steps that execute while the request is in flight, then
exponentially decaying soft weights. Guidance acts on the predicted clean
chunk ``x0 = x_t - t * v`` and steps with the velocity that reaches the
blended estimate, so masked rows converge to the previous chunk while the
rest stays conditioned on the new images. Without a previous chunk this is
exactly the default Euler sampler. The loop is the wrapper's
``_sample_actions_from_prefix_cache``, shared with Stage2's reference chunk.
"""

import torch

from rlinf.models.embodiment.openpi.rtc_guidance import build_rtc_target_and_mask


@torch.no_grad()
def sample_actions_rtc(wrapper, observation, rtc_context=None, *, rng=None, hard_only=False):
    """Return model-space actions [B, H, A_model] and a guidance summary.

    ``hard_only`` keeps only the in-flight rows of the previous chunk, e.g.
    after a subtask switch, so the old plan does not pull the new one.
    """
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import (
        preprocess_observation,
    )

    model = wrapper.model
    observation = preprocess_observation(observation, train=False)
    batch = observation.state.shape[0]
    device = observation.state.device
    noise = torch.randn(
        batch, model.action_horizon, model.action_dim, device=device, generator=rng
    )
    _, prefix_mask, kv_cache = model.build_prefix_cache(observation)

    prev = rtc_context.get_prev_remaining() if rtc_context is not None else None
    delay = 0 if rtc_context is None else int(rtc_context.delay_steps)
    if prev is not None:
        prev = prev.to(device=device, dtype=noise.dtype)
        if hard_only:
            prev = prev[:, :delay]
    target, mask = build_rtc_target_and_mask(
        prev_remaining=prev,
        horizon=model.action_horizon,
        action_dim=model.action_dim,
        delay_steps=delay,
        device=device,
        dtype=noise.dtype,
    )
    guided = prev is not None and prev.numel() > 0
    x_t = wrapper._sample_actions_from_prefix_cache(
        observation,
        prefix_mask,
        kv_cache,
        noise=noise,
        rtc_target=target if guided else None,
        rtc_mask=mask if guided else None,
    )

    summary = {"guided": bool(guided)}
    if guided:
        weights = mask[0, :, 0]
        summary.update(
            executed_horizon=int(rtc_context.executed_horizon),
            delay_steps=delay,
            overlap_steps=int(min(prev.shape[1], model.action_horizon)),
            hard_steps=int((weights >= 1.0).sum().item()),
            soft_steps=int(((weights > 0) & (weights < 1.0)).sum().item()),
            hard_only=bool(hard_only),
        )
    return x_t, summary
