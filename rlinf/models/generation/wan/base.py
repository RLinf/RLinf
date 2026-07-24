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
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.generation.sd3.utils import prompt_list


@dataclass
class Wan22Config:
    model_path: str = ""
    condition_mode: Literal["t2v", "ti2v"] = "t2v"
    resolution: list[int] = field(default_factory=lambda: [480, 480])
    num_frames: int = 1
    num_steps: int = 8
    eval_num_steps: int = 20
    timestep_fraction: float = 0.99
    guidance_scale: float = 1.0
    eval_guidance_scale: float = 1.0
    cfg: bool = False
    max_sequence_length: int = 512
    output_type: str = "pt"
    rl_mode: Literal["flow-grpo", "nft"] = "nft"
    weight_format: Literal["diffusers", "vidar"] = "diffusers"
    vidar_path: str | None = None
    use_lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_path: str | None = None
    init_lora_weights: str = "gaussian"
    compile_transformer_forward: bool = False
    compile_mode: str = "default"
    max_generation_batch_size: int = 0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "attn1.to_q",
            "attn1.to_k",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_q",
            "attn2.to_k",
            "attn2.to_v",
            "attn2.to_out.0",
            "ffn.net.0.proj",
            "ffn.net.2",
        ]
    )
    offload_auxiliary_modules: bool = False
    enable_vae_tiling: bool = False


class Wan22Model(torch.nn.Module, BasePolicy):
    """Shared Wan 2.2 video policy logic for RL/NFT rollout."""

    def __init__(self, config: Wan22Config, pipeline: Any):
        super().__init__()
        self.config = config
        self.model_path = str(config.model_path)
        self.pipeline = pipeline
        self.transformer = pipeline.transformer
        self._compiled_transformer_forward = None

    @property
    def _no_split_modules(self) -> list[str]:
        return list(
            getattr(self.transformer, "_no_split_modules", ["WanTransformerBlock"])
        )

    def to(self, device):
        module = super().to(device)
        device = torch.device(device)
        if device.type == "cpu":
            self.pipeline.vae.to(device=device)
            self.pipeline.text_encoder.to(device=device)
            if getattr(self.pipeline, "image_encoder", None) is not None:
                self.pipeline.image_encoder.to(device=device)
        return module

    def obs_processor(self, env_obs: Any) -> tuple[list[str], dict[str, Any]]:
        raise NotImplementedError

    def nft_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        nft_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        raise NotImplementedError

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
        raise NotImplementedError

    @staticmethod
    def _configure_torch_dynamo_for_compile() -> None:
        dynamo_config = torch._dynamo.config
        for name, value in (
            ("cache_size_limit", 1000),
            ("recompile_limit", 800),
            ("accumulated_cache_size_limit", 1000),
            ("accumulated_recompile_limit", 2000),
        ):
            if hasattr(dynamo_config, name):
                setattr(dynamo_config, name, max(getattr(dynamo_config, name), value))

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.NFT:
            return self.nft_forward(**kwargs)
        raise ValueError(f"Unknown forward_type: {forward_type}")

    def default_forward(self, forward_inputs: dict[str, torch.Tensor], **kwargs):
        del kwargs
        batch_size = forward_inputs["nft_x0"].shape[0]
        num_train_timesteps = min(
            max(1, int(self.config.num_steps * self.config.timestep_fraction)),
            self.config.num_steps,
        )
        return {
            "logprobs": torch.zeros(
                batch_size,
                num_train_timesteps,
                device=forward_inputs["nft_x0"].device,
                dtype=forward_inputs["nft_x0"].dtype,
            ),
            "values": None,
        }

    def _prepare_nft_forward_inputs(
        self,
        forward_inputs: dict[str, torch.Tensor],
        nft_inputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor | torch.device | torch.dtype]:
        parameter = next(self.transformer.parameters())
        device = parameter.device
        model_dtype = parameter.dtype
        x_t = nft_inputs["x_t"].to(device=device, dtype=model_dtype)
        model_x_t = x_t.movedim(1, 2)
        timesteps = self._scheduler_to_model_timesteps(
            nft_inputs["timesteps"].to(device=device),
            model_x_t,
        )
        prompt_embeds = forward_inputs["prompt_embeds"].to(
            device=device,
            dtype=model_dtype,
        )
        negative_prompt_embeds = forward_inputs.get("negative_prompt_embeds")
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(
                device=device,
                dtype=model_dtype,
            )
        return {
            "device": device,
            "model_dtype": model_dtype,
            "x_t": x_t,
            "model_x_t": model_x_t,
            "timesteps": timesteps,
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

    @torch.no_grad()
    def encode_prompts(
        self,
        prompts: str | Sequence[str],
        *,
        negative_prompts: str | Sequence[str],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        device = next(self.transformer.parameters()).device
        self.pipeline.text_encoder.to(device=device)
        prompt_embeds, negative_prompt_embeds = self.pipeline.encode_prompt(
            prompt=prompt_list(prompts),
            negative_prompt=prompt_list(negative_prompts),
            do_classifier_free_guidance=self.config.cfg,
            num_videos_per_prompt=1,
            max_sequence_length=self.config.max_sequence_length,
            device=device,
            dtype=next(self.transformer.parameters()).dtype,
        )
        negative_prompt_embeds_out = None
        if self.config.cfg:
            negative_prompt_embeds_out = negative_prompt_embeds.to(device)
        return prompt_embeds.to(device), negative_prompt_embeds_out

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del compute_values
        prompts, conditions = self.obs_processor(env_obs)
        negative_prompts = kwargs.get("negative_prompts") or [""] * len(prompts)
        prompt_embeds, negative_prompt_embeds = self.encode_prompts(
            prompts,
            negative_prompts=negative_prompts,
        )
        if self.config.offload_auxiliary_modules:
            self.pipeline.text_encoder.to(device="cpu")
            torch.cuda.empty_cache()

        is_eval = mode == "eval"
        if is_eval:
            num_steps = self.config.eval_num_steps
            guidance_scale = self.config.eval_guidance_scale
        else:
            num_steps = self.config.num_steps
            guidance_scale = self.config.guidance_scale
        images, final_latents, denoise_info = self._denoise_batched(
            conditions=conditions,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            num_steps=num_steps,
            generator=kwargs.get("generator"),
            latents=kwargs.get("latents"),
        )
        if is_eval:
            return images, {"prev_values": None}

        batch_size = final_latents.shape[0]
        num_train_timesteps = min(
            max(1, int(self.config.num_steps * self.config.timestep_fraction)),
            num_steps,
        )
        old_logprobs = torch.zeros(
            batch_size,
            num_train_timesteps,
            device=final_latents.device,
            dtype=final_latents.dtype,
        )
        forward_inputs = {
            "nft_x0": final_latents.movedim(2, 1).detach(),
            "nft_noise_level": torch.zeros(
                final_latents.shape[0],
                device=final_latents.device,
                dtype=final_latents.dtype,
            ),
            "prompt_embeds": prompt_embeds.detach(),
        }
        if self.config.cfg:
            forward_inputs["negative_prompt_embeds"] = negative_prompt_embeds.detach()
        for key, value in denoise_info.items():
            forward_inputs[key] = value.detach()

        return images, {
            "prev_logprobs": old_logprobs,
            "prev_values": None,
            "forward_inputs": forward_inputs,
        }

    @torch.no_grad()
    def _denoise_batched(
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
        max_batch = int(self.config.max_generation_batch_size)
        if max_batch <= 0 or prompt_embeds.shape[0] <= max_batch:
            return self._denoise(
                conditions=conditions,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                generator=generator,
                latents=latents,
            )

        image_chunks = []
        latent_chunks = []
        denoise_info_chunks = []
        for start in range(0, prompt_embeds.shape[0], max_batch):
            end = start + max_batch
            chunk_negative_prompt_embeds = (
                negative_prompt_embeds[start:end] if self.config.cfg else None
            )
            chunk_latents = latents[start:end] if latents is not None else None
            chunk_conditions = {
                key: value[start:end] for key, value in conditions.items()
            }
            images, final_latents, denoise_info = self._denoise(
                conditions=chunk_conditions,
                prompt_embeds=prompt_embeds[start:end],
                negative_prompt_embeds=chunk_negative_prompt_embeds,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                generator=generator,
                latents=chunk_latents,
            )
            image_chunks.append(images)
            latent_chunks.append(final_latents)
            denoise_info_chunks.append(denoise_info)
        denoise_info = {}
        if denoise_info_chunks:
            for key in denoise_info_chunks[0]:
                denoise_info[key] = torch.cat(
                    [chunk[key] for chunk in denoise_info_chunks],
                    dim=0,
                )
        return (
            torch.cat(image_chunks, dim=0),
            torch.cat(latent_chunks, dim=0),
            denoise_info,
        )

    def _prepare_denoise_timesteps(
        self,
        num_steps: int,
        device: torch.device,
    ) -> torch.Tensor:
        self.pipeline.scheduler.set_timesteps(num_steps, device=device)
        if hasattr(self.pipeline.scheduler, "sigmas"):
            self.pipeline.scheduler.sigmas = self.pipeline.scheduler.sigmas.to(device)
        return self.pipeline.scheduler.timesteps

    def _call_transformer(self, **kwargs):
        if not self.config.compile_transformer_forward:
            return self.transformer(**kwargs)
        if self._compiled_transformer_forward is None:
            self._configure_torch_dynamo_for_compile()
            self._compiled_transformer_forward = torch.compile(
                self.transformer.forward,
                mode=self.config.compile_mode,
            )
        return self._compiled_transformer_forward(**kwargs)

    def _scheduler_to_model_timesteps(
        self,
        timesteps: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        timesteps = timesteps.reshape(-1).to(device=latents.device)
        if timesteps.dtype.is_floating_point and timesteps.max() <= 1.0:
            timesteps = timesteps.to(dtype=torch.float32) * 1000.0
        if not getattr(self.pipeline.config, "expand_timesteps", False):
            return timesteps.expand(latents.shape[0])

        if timesteps.numel() == 1:
            timesteps = timesteps.expand(latents.shape[0])
        mask = torch.ones(latents.shape, dtype=torch.float32, device=latents.device)
        expanded = mask[:, 0, :, ::2, ::2] * timesteps.to(torch.float32).view(
            -1, 1, 1, 1
        )
        return expanded.flatten(1)

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor, output_type: str = "pt"):
        self.pipeline.vae.to(device=latents.device)
        latents = latents.to(dtype=self.pipeline.vae.dtype)
        latents_mean = (
            torch.tensor(self.pipeline.vae.config.latents_mean)
            .view(1, self.pipeline.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.pipeline.vae.config.latents_std).view(
            1, self.pipeline.vae.config.z_dim, 1, 1, 1
        ).to(latents.device, latents.dtype)
        latents = latents / latents_std + latents_mean
        if self.config.enable_vae_tiling:
            self.pipeline.vae.enable_tiling()
        video = self.pipeline.vae.decode(latents, return_dict=False)[0]
        return self.pipeline.video_processor.postprocess_video(video, output_type)

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
        wan_config = dict(config)
        wan_config.pop("model_path", None)
        return {"model_path": self.model_path, "wan22_ti2v_5b": wan_config}


__all__ = ["Wan22Config", "Wan22Model"]
