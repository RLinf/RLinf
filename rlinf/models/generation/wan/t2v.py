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

from typing import Any

import torch

from rlinf.models.generation.sd3.utils import prompt_list
from rlinf.models.generation.wan.base import Wan22Model


class Wan22T2VModel(Wan22Model):
    """Wan 2.2 text-to-video model."""

    def obs_processor(self, env_obs: Any) -> tuple[list[str], dict[str, Any]]:
        return prompt_list(env_obs["task_descriptions"]), {}

    def nft_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        nft_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs
        nft_batch = self._prepare_nft_forward_inputs(
            forward_inputs,
            nft_inputs,
        )
        v_theta = self._transformer_forward(
            nft_batch["model_x_t"],
            nft_batch["timesteps"],
            nft_batch["prompt_embeds"],
            nft_batch["negative_prompt_embeds"],
            self.config.guidance_scale,
        )
        v_theta = v_theta.movedim(2, 1)
        return {"v_theta": v_theta}

    @torch.no_grad()
    def _denoise(
        self,
        *,
        conditions: dict[str, Any],
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
        num_steps: int,
        generator=None,
        latents=None,
    ):
        del conditions
        device = prompt_embeds.device
        model_dtype = next(self.transformer.parameters()).dtype
        timesteps = self._prepare_denoise_timesteps(num_steps, device)
        height, width = self.config.resolution
        if self.config.cfg:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=device,
                dtype=model_dtype,
            )
        latents = self.pipeline.prepare_latents(
            batch_size=prompt_embeds.shape[0],
            num_channels_latents=self.transformer.config.in_channels,
            height=height,
            width=width,
            num_frames=self.config.num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        self.pipeline._num_timesteps = len(timesteps)
        for t in timesteps:
            model_latents = latents.to(dtype=model_dtype)
            timestep = self._scheduler_to_model_timesteps(
                t.expand(latents.shape[0]),
                model_latents,
            )
            noise_pred = self._transformer_forward(
                model_latents,
                timestep,
                prompt_embeds.to(dtype=model_dtype),
                negative_prompt_embeds,
                guidance_scale,
            )
            latents = self.pipeline.scheduler.step(
                noise_pred.float(),
                t,
                latents.float(),
                return_dict=False,
            )[0]
        videos = self.decode_latents(latents, output_type=self.config.output_type)
        if self.config.num_frames == 1:
            images = videos[:, 0]
        else:
            images = videos
        if self.config.offload_auxiliary_modules:
            self.pipeline.vae.to(device="cpu")
            torch.cuda.empty_cache()
        return images, latents, {}

    def _transformer_forward(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
    ) -> torch.Tensor:
        noise_pred = self._call_transformer(
            hidden_states=latents,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        if self.config.cfg:
            noise_uncond = self._call_transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        return noise_pred


__all__ = ["Wan22T2VModel"]
