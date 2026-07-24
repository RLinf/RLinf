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

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from rlinf.models.generation.sd3.sampler import (
    init_solver_state,
    model_timestep,
    sample_latents_step,
)


def prompt_list(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def move_text_encoders(
    pipeline: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
):
    kwargs = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype

    pipeline.text_encoder.to(**kwargs)
    pipeline.text_encoder_2.to(**kwargs)
    pipeline.text_encoder_3.to(**kwargs)


def move_auxiliary_modules(
    pipeline: Any,
    *,
    device: torch.device | str | None = None,
):
    pipeline.vae.to(device=device)
    move_text_encoders(pipeline, device=device)


def configure_pipeline_trainability(
    pipeline: Any,
    *,
    train_transformer: bool,
):
    pipeline.set_progress_bar_config(disable=True)

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(train_transformer)


def move_pipeline_modules(pipeline: Any, *, inference_dtype: torch.dtype):
    pipeline.vae.to(dtype=torch.float32)
    move_text_encoders(pipeline, dtype=inference_dtype)
    pipeline.transformer.to(dtype=inference_dtype)


@torch.no_grad()
def denoise_with_logprob(
    *,
    pipeline: Any,
    transformer: torch.nn.Module,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor | None,
    negative_pooled_prompt_embeds: torch.Tensor | None,
    cfg_enabled: bool,
    guidance_scale: float,
    noise_level: float,
    num_steps: int,
    resolution: int,
    output_type: str,
    solver: str = "sde",
    offload_vae: bool = False,
    return_trajectory: bool = True,
    generator=None,
    latents=None,
):
    pipeline._guidance_scale = guidance_scale
    pipeline._interrupt = False

    model_prompt_embeds = prompt_embeds
    model_pooled_embeds = pooled_prompt_embeds
    if cfg_enabled:
        model_prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        model_pooled_embeds = torch.cat(
            [negative_pooled_prompt_embeds, pooled_prompt_embeds],
            dim=0,
        )

    latents = pipeline.prepare_latents(
        prompt_embeds.shape[0],
        transformer.config.in_channels,
        resolution,
        resolution,
        prompt_embeds.dtype,
        prompt_embeds.device,
        generator,
        latents,
    ).float()
    timesteps, num_steps = retrieve_timesteps(
        pipeline.scheduler,
        num_steps,
        prompt_embeds.device,
    )
    pipeline._num_timesteps = len(timesteps)
    model_dtype = next(transformer.parameters()).dtype
    sigmas = pipeline.scheduler.sigmas.to(
        device=prompt_embeds.device,
        dtype=torch.float32,
    )
    solver, dpm_state = init_solver_state(solver)
    use_dpm = dpm_state is not None

    latent_chain = [latents] if return_trajectory else None
    log_probs = [] if return_trajectory else None
    with pipeline.progress_bar(total=num_steps) as progress_bar:
        for step_index, t in enumerate(timesteps):
            model_latents = latents.to(dtype=model_dtype)
            model_input = (
                torch.cat([model_latents] * 2) if cfg_enabled else model_latents
            )
            timestep = model_timestep(
                t,
                sigmas=sigmas,
                step_index=step_index,
                batch_size=model_input.shape[0],
                use_dpm=use_dpm,
            )
            noise_pred = transformer(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=model_prompt_embeds,
                pooled_projections=model_pooled_embeds,
                return_dict=False,
            )[0]
            if cfg_enabled:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            latents, log_prob = sample_latents_step(
                scheduler=pipeline.scheduler,
                solver=solver,
                noise_pred=noise_pred,
                timestep=t,
                latents=latents,
                noise_level=noise_level,
                compute_logprob=return_trajectory,
                step_index=step_index,
                sigmas=sigmas,
                dpm_state=dpm_state,
            )
            if return_trajectory:
                latent_chain.append(latents)
                log_probs.append(log_prob)
            progress_bar.update()

    pipeline.vae.to(device=latents.device)
    latents = (
        latents / pipeline.vae.config.scaling_factor
    ) + pipeline.vae.config.shift_factor
    latents = latents.to(dtype=pipeline.vae.dtype)
    images = pipeline.vae.decode(latents, return_dict=False)[0]
    images = pipeline.image_processor.postprocess(images, output_type=output_type)
    pipeline.maybe_free_model_hooks()

    if offload_vae:
        pipeline.vae.to(device="cpu")
        torch.cuda.empty_cache()

    return images, latent_chain, log_probs
