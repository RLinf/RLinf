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

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.generation.sd3.sampler import sde_step_with_logprob
from rlinf.models.generation.sd3.utils import (
    denoise_with_logprob,
    move_auxiliary_modules,
    move_text_encoders,
    prompt_list,
)


@dataclass
class StableDiffusion3Config:
    model_path: str = ""
    resolution: int = 512
    num_steps: int = 10
    eval_num_steps: int = 40
    timestep_fraction: float = 0.99
    guidance_scale: float = 4.5
    eval_guidance_scale: float = 4.5
    cfg: bool = True
    noise_level: float = 0.7
    eval_noise_level: float = 0.0
    train_sampler: Literal["sde", "ode", "dpm1", "dpm2"] = "sde"
    eval_sampler: Literal["sde", "ode", "dpm1", "dpm2"] = "ode"
    max_sequence_length: int = 128
    output_type: str = "pt"
    rl_mode: Literal["flow-grpo", "nft"] = "flow-grpo"
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_path: str | None = None
    init_lora_weights: str = "gaussian"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
    )
    offload_auxiliary_modules: bool = True

    def update_from_dict(self, config_dict: Mapping[str, Any] | None):
        if not config_dict:
            return
        unknown_fields = sorted(set(config_dict) - set(self.__dataclass_fields__))
        if unknown_fields:
            raise ValueError(
                f"Unknown StableDiffusion3 config fields: {unknown_fields}"
            )
        for key, value in config_dict.items():
            setattr(self, key, value)


class StableDiffusion3(torch.nn.Module, BasePolicy):
    """Stable Diffusion 3 policy model for image rollout and logprob updates."""

    def __init__(self, config: StableDiffusion3Config, pipeline: Any):
        super().__init__()
        self.config = config
        self.model_path = str(config.model_path)
        self.pipeline = pipeline
        self.transformer = pipeline.transformer

    @property
    def _no_split_modules(self) -> list[str]:
        return list(
            getattr(self.transformer, "_no_split_modules", ["JointTransformerBlock"])
        )

    def to(self, device):
        module = super().to(device)
        device = torch.device(device)
        if device.type == "cpu":
            move_auxiliary_modules(self.pipeline, device=device)
        return module

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        prompt_embeds = forward_inputs["prompt_embeds"]
        pooled_prompt_embeds = forward_inputs["pooled_prompt_embeds"]
        retrieve_timesteps(
            self.pipeline.scheduler,
            self.config.num_steps,
            prompt_embeds.device,
        )
        if self.config.cfg:
            prompt_embeds = torch.cat(
                [forward_inputs["negative_prompt_embeds"], prompt_embeds],
                dim=0,
            )
            pooled_prompt_embeds = torch.cat(
                [forward_inputs["negative_pooled_prompt_embeds"], pooled_prompt_embeds],
                dim=0,
            )

        logprobs = []
        model_dtype = next(self.transformer.parameters()).dtype
        num_train_timesteps = forward_inputs["latents"].shape[1]
        for step_index in range(num_train_timesteps):
            latents = forward_inputs["latents"][:, step_index]
            timesteps = forward_inputs["timesteps"][:, step_index]
            model_latents = latents.to(dtype=model_dtype)
            model_input = (
                torch.cat([model_latents] * 2) if self.config.cfg else model_latents
            )
            timestep = torch.cat([timesteps] * 2) if self.config.cfg else timesteps
            noise_pred = self.transformer(
                hidden_states=model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
            if self.config.cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.config.guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            _, logprob, _, _ = sde_step_with_logprob(
                self.pipeline.scheduler,
                noise_pred,
                timesteps,
                latents,
                noise_level=self.config.noise_level,
                prev_sample=forward_inputs["next_latents"][:, step_index],
            )
            logprobs.append(logprob)

        return {"logprobs": torch.stack(logprobs, dim=1), "values": None}

    def nft_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        nft_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        """Predict velocity for explicitly provided NFT noisy latents."""
        device = next(self.transformer.parameters()).device
        model_dtype = next(self.transformer.parameters()).dtype
        x_t = nft_inputs["x_t"].to(device)
        timesteps = nft_inputs["timesteps"].to(device)

        prompt_embeds = forward_inputs["prompt_embeds"].to(device)
        pooled_prompt_embeds = forward_inputs["pooled_prompt_embeds"].to(device)
        model_input = x_t[:, 0].to(dtype=model_dtype)
        model_timesteps = timesteps
        if model_timesteps.dtype.is_floating_point and model_timesteps.max() <= 1.0:
            model_timesteps = model_timesteps.to(dtype=torch.float32) * 1000.0
        if self.config.cfg:
            prompt_embeds = torch.cat(
                [forward_inputs["negative_prompt_embeds"].to(device), prompt_embeds],
                dim=0,
            )
            pooled_prompt_embeds = torch.cat(
                [
                    forward_inputs["negative_pooled_prompt_embeds"].to(device),
                    pooled_prompt_embeds,
                ],
                dim=0,
            )
            model_input = torch.cat([model_input] * 2)
            model_timesteps = torch.cat([model_timesteps] * 2)

        v_theta = self.transformer(
            hidden_states=model_input,
            timestep=model_timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        if self.config.cfg:
            v_uncond, v_text = v_theta.chunk(2)
            v_theta = v_uncond + self.config.guidance_scale * (v_text - v_uncond)
        v_theta = v_theta[:, None]

        return {"v_theta": v_theta}

    def obs_processor(self, env_obs: Any) -> list[str]:
        if isinstance(env_obs, Mapping):
            for key in ("prompts", "prompt", "texts", "text", "task_descriptions"):
                if key in env_obs:
                    return prompt_list(env_obs[key])
        if isinstance(env_obs, Sequence) and not isinstance(env_obs, str):
            return [str(prompt) for prompt in env_obs]
        return [str(env_obs)]

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: str | Sequence[str],
        *,
        num_images_per_prompt: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.transformer.parameters()).device
        move_text_encoders(self.pipeline, device=device)
        prompt_embeds, _, pooled_prompt_embeds, _ = self.pipeline.encode_prompt(
            prompt=prompt_list(prompts),
            prompt_2=None,
            prompt_3=None,
            do_classifier_free_guidance=False,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=self.config.max_sequence_length,
        )
        return prompt_embeds.to(device), pooled_prompt_embeds.to(device)

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        prompts = self.obs_processor(env_obs)
        prompt_embeds, pooled_prompt_embeds = self.encode_prompts(prompts)
        negative_prompt_embeds, negative_pooled_prompt_embeds = None, None
        if self.config.cfg:
            negative_prompts = kwargs.get("negative_prompts") or [""] * len(prompts)
            negative_prompt_embeds, negative_pooled_prompt_embeds = self.encode_prompts(
                negative_prompts
            )
        if self.config.offload_auxiliary_modules:
            move_text_encoders(self.pipeline, device="cpu")
            torch.cuda.empty_cache()

        is_eval = mode == "eval"
        num_steps = (
            self.config.eval_num_steps if mode == "eval" else self.config.num_steps
        )
        guidance_scale = (
            self.config.eval_guidance_scale
            if mode == "eval"
            else self.config.guidance_scale
        )
        noise_level = (
            self.config.eval_noise_level if mode == "eval" else self.config.noise_level
        )
        solver = self.config.eval_sampler if is_eval else self.config.train_sampler
        images, latents_chain, log_probs = denoise_with_logprob(
            pipeline=self.pipeline,
            transformer=self.transformer,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            cfg_enabled=self.config.cfg,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            solver=solver,
            num_steps=num_steps,
            resolution=self.config.resolution,
            output_type=self.config.output_type,
            offload_vae=self.config.offload_auxiliary_modules,
            return_trajectory=not is_eval,
            generator=kwargs.get("generator"),
            latents=kwargs.get("latents"),
        )
        if is_eval:
            return images, {"prev_values": None}

        timesteps = self.pipeline.scheduler.timesteps[: len(log_probs)].repeat(
            log_probs[0].shape[0],
            1,
        )
        full_latents = torch.stack(latents_chain, dim=1)
        old_logprobs = torch.stack(log_probs, dim=1)
        num_train_timesteps = min(
            max(1, int(self.config.num_steps * self.config.timestep_fraction)),
            old_logprobs.shape[1],
        )

        forward_inputs = {
            "latents": full_latents[:, :num_train_timesteps].detach(),
            "next_latents": full_latents[:, 1 : 1 + num_train_timesteps].detach(),
            "timesteps": timesteps[:, :num_train_timesteps].detach(),
            "prompt_embeds": prompt_embeds.detach(),
            "pooled_prompt_embeds": pooled_prompt_embeds.detach(),
        }
        if negative_prompt_embeds is not None:
            forward_inputs["negative_prompt_embeds"] = negative_prompt_embeds.detach()
            forward_inputs["negative_pooled_prompt_embeds"] = (
                negative_pooled_prompt_embeds.detach()
            )
        if self.config.rl_mode == "nft":
            forward_inputs["nft_x0"] = full_latents[:, -1][:, None].detach()
            forward_inputs["nft_noise_level"] = torch.zeros(
                full_latents.shape[0],
                device=full_latents.device,
                dtype=full_latents.dtype,
            )

        return images, {
            "prev_logprobs": old_logprobs[:, :num_train_timesteps].detach(),
            "prev_values": None,
            "forward_inputs": forward_inputs,
        }

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        enable_fn = getattr(self.transformer, "gradient_checkpointing_enable", None)
        if enable_fn is None:
            return
        if gradient_checkpointing_kwargs is None:
            enable_fn()
        else:
            enable_fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        disable_fn = getattr(self.transformer, "gradient_checkpointing_disable", None)
        if disable_fn is not None:
            disable_fn()

    def trainable_parameters(self):
        return [param for param in self.parameters() if param.requires_grad]

    def export_config(self) -> dict[str, Any]:
        config = asdict(self.config)
        sd3_config = dict(config)
        sd3_config.pop("model_path", None)
        return {"model_path": self.model_path, "sd3": sd3_config}
