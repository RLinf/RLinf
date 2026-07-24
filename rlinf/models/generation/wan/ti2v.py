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

import numpy as np
import torch
from PIL import Image

from rlinf.models.generation.sd3.utils import prompt_list
from rlinf.models.generation.wan.base import Wan22Model


class Wan22TI2VModel(Wan22Model):
    """Wan 2.2 text-image-to-video model."""

    def obs_processor(self, env_obs: Any) -> tuple[list[str], dict[str, Any]]:
        prompts = prompt_list(env_obs["task_descriptions"])
        main_images = env_obs["main_images"]
        if torch.is_tensor(main_images):
            main_images = main_images.detach().cpu().numpy()
        if isinstance(main_images, list):
            main_images = np.concatenate(main_images, axis=0)
        main_images = [Image.fromarray(image.astype(np.uint8)) for image in main_images]
        return prompts, {"image_condition": main_images}

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
        latent_condition = forward_inputs["latent_condition"].to(
            device=nft_batch["device"],
            dtype=nft_batch["model_dtype"],
        )
        first_frame_mask = forward_inputs["first_frame_mask"].to(
            device=nft_batch["device"],
            dtype=nft_batch["model_dtype"],
        )
        model_x_t = (
            1 - first_frame_mask
        ) * latent_condition + first_frame_mask * nft_batch["model_x_t"]
        timestep_mask = first_frame_mask[0, 0, :, ::2, ::2].flatten().unsqueeze(0)
        v_theta = self._transformer_forward_i2v(
            model_x_t,
            nft_batch["timesteps"] * timestep_mask,
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
        image_condition = conditions["image_condition"]
        device = prompt_embeds.device
        model_dtype = next(self.transformer.parameters()).dtype
        height, width = self.config.resolution
        if self.config.cfg:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=device,
                dtype=model_dtype,
            )
        timesteps = self._prepare_denoise_timesteps(num_steps, device)

        self.pipeline.vae.to(device=device)
        image_tensor = self.pipeline.video_processor.preprocess(
            image_condition,
            height=height,
            width=width,
        ).to(device=device, dtype=torch.float32)
        latent_outputs = self.pipeline.prepare_latents(
            image_tensor,
            batch_size=image_tensor.shape[0],
            num_channels_latents=self.pipeline.vae.config.z_dim,
            height=height,
            width=width,
            num_frames=self.config.num_frames,
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=latents,
        )
        latents, latent_condition, first_frame_mask = latent_outputs
        if latent_condition.shape[0] != latents.shape[0]:
            latent_condition = latent_condition[: latents.shape[0]]
        first_frame_mask = first_frame_mask.expand(
            latents.shape[0],
            *first_frame_mask.shape[1:],
        ).contiguous()

        self.pipeline._num_timesteps = len(timesteps)
        for t in timesteps:
            latent_model_input = (
                (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
            ).to(dtype=model_dtype)
            timestep = (first_frame_mask[0, 0, :, ::2, ::2] * t).flatten().unsqueeze(0)
            timestep = timestep.expand(latents.shape[0], -1)
            noise_pred = self._transformer_forward_i2v(
                latent_model_input,
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
        latents = (1 - first_frame_mask) * latent_condition + first_frame_mask * latents
        videos = self.decode_latents(latents, output_type=self.config.output_type)
        if self.config.num_frames == 1:
            images = videos[:, 0]
        else:
            images = videos
        if self.config.offload_auxiliary_modules:
            self.pipeline.vae.to(device="cpu")
            torch.cuda.empty_cache()
        denoise_info = {
            "latent_condition": latent_condition,
            "first_frame_mask": first_frame_mask,
        }
        return images, latents, denoise_info

    def _transformer_forward_i2v(
        self,
        latents_with_condition: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: torch.Tensor | None,
        guidance_scale: float,
    ) -> torch.Tensor:
        noise_pred = self._call_transformer(
            hidden_states=latents_with_condition,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
        )[0]
        if self.config.cfg:
            noise_uncond = self._call_transformer(
                hidden_states=latents_with_condition,
                timestep=timestep,
                encoder_hidden_states=negative_prompt_embeds,
                return_dict=False,
            )[0]
            noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
        return noise_pred


__all__ = ["Wan22TI2VModel"]
