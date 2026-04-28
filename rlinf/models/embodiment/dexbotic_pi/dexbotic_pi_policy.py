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

import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy
from dexbotic.model.pi0.pi0_arch import (
    Pi0ForCausalLM,
    make_attn_mask,
    make_attn_mask_4d,
)
from PIL import Image
from transformers import DynamicCache

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.utils.logging import get_logger


class DexboticPi0ForRLActionPrediction(BasePolicy, Pi0ForCausalLM):
    _no_split_names = [
        "action_in_proj",
        "action_out_proj",
        "state_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]

    def __init__(self, config):
        Pi0ForCausalLM.__init__(self, config)
        if getattr(config, "train_expert_only", False):
            self._no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        else:
            self._no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
            ]
        self.logger = get_logger()
        model_dtype = None
        if (
            hasattr(self.model, "llm")
            and hasattr(self.model.llm, "layers")
            and len(self.model.llm.layers) > 0
        ):
            first_layer = self.model.llm.layers[0]
            for param in first_layer.parameters():
                model_dtype = param.dtype
                break
        elif (
            hasattr(self.model, "action_expert")
            and hasattr(self.model.action_expert, "layers")
            and len(self.model.action_expert.layers) > 0
        ):
            first_layer = self.model.action_expert.layers[0]
            for param in first_layer.parameters():
                model_dtype = param.dtype
                break
        if model_dtype is None:
            all_params = list(self.model.parameters())
            if all_params:
                model_dtype = all_params[0].dtype
            else:
                model_dtype = torch.float32
        self.model = self.model.to(dtype=model_dtype)

        if hasattr(self.model, "action_expert") and hasattr(
            self.model.action_expert, "embed_tokens"
        ):
            self.model.action_expert.embed_tokens = None
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

        self.config = config
        self.num_steps = config.num_steps
        self.action_horizon = config.chunk_size
        self.num_action_chunks = getattr(
            config, "output_action_chunks", config.chunk_size
        )
        self.action_dim = config.action_dim
        self.non_delta_mask = getattr(
            config, "non_delta_mask", [6]
        )  # Indices of absolute actions (default: [6] for gripper)
        self.global_step = 0
        self.use_vlm_value = False
        self.value_head = nn.Linear(config.action_config.hidden_size, 1)
        self.value_head = self.value_head.to(
            dtype=self.model.action_out_proj.weight.dtype
        )
        self._input_transform = None
        self._output_transform = None
        self.norm_stats = None
        self.pi0_tokenization = None

    def freeze_vlm(self):
        if not getattr(self.config, "train_expert_only", False):
            self.logger.warning("freeze_vlm() called but train_expert_only is False")
            return
        # Freeze vision tower
        if getattr(self.model, "mm_vision_tower", None) is not None:
            self.model.mm_vision_tower.eval()
            for param in self.model.mm_vision_tower.parameters():
                param.requires_grad = False
        # Freeze LLM
        if getattr(self.model, "llm", None) is not None:
            self.model.llm.eval()
            for param in self.model.llm.parameters():
                param.requires_grad = False
        # Freeze mm_projector
        if getattr(self.model, "mm_projector", None) is not None:
            self.model.mm_projector.eval()
            for param in self.model.mm_projector.parameters():
                param.requires_grad = False

    def _read_normalization_stats(self, norm_stats_file):
        if not os.path.exists(norm_stats_file):
            raise FileNotFoundError(
                f"Normalization stats not found at {norm_stats_file}. "
                "Make sure the checkpoint directory contains norm_stats.json"
            )
        with open(norm_stats_file, "r") as f:
            norm_stats = json.load(f)
            if "norm_stats" in norm_stats:
                norm_stats = norm_stats["norm_stats"]
        return ToNumpy()(norm_stats)

    def setup_wrappers(self, transforms=(), output_transforms=()):
        if transforms:
            self._input_transform = Pipeline(transforms)
        else:
            self._input_transform = None

        if output_transforms:
            self._output_transform = Pipeline(output_transforms)
        else:
            self._output_transform = None

    def input_transform(self, obs: dict, transpose=True):
        if "prompt" in obs:
            prompts = obs["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts]
            elif isinstance(prompts, torch.Tensor):
                prompts = [str(p) for p in prompts]
            batch_input_ids = []
            for prompt in prompts:
                tokenized = self.pi0_tokenization([{"value": prompt}])
                batch_input_ids.append(tokenized["input_ids"])

            batch_input_ids = torch.from_numpy(np.array(batch_input_ids))
            batch_attention_mask = batch_input_ids != self.tokenizer.pad_token_id

            obs["tokenized_prompt"] = batch_input_ids
            obs["tokenized_prompt_mask"] = batch_attention_mask

        if self._input_transform is not None:
            if "observation/state" in obs:
                state_tensor = obs["observation/state"]
                if isinstance(state_tensor, torch.Tensor):
                    state_value = state_tensor.cpu().float().numpy()
                else:
                    state_value = state_tensor

                state_dict = {"state": state_value}
                state_dict = self._input_transform(state_dict)

                obs["observation/state"] = state_dict["state"]
                obs["states"] = state_dict["state"]
        return obs

    def output_transform(self, outputs):
        if self._output_transform is None:
            self.logger.warning(
                "[output_transform] WARNING: _output_transform is None! Actions will NOT be denormalized!"
            )
            return outputs

        state_batch = outputs.get("state", None)
        meta_data = outputs.get("meta_data", {})

        batch_size = outputs["actions"].shape[0]
        transformed_actions = []

        for i in range(batch_size):
            sample = {"action": outputs["actions"][i].cpu().numpy()}
            if state_batch is not None:
                if isinstance(state_batch, torch.Tensor):
                    sample["state"] = state_batch[i].cpu().numpy()
                else:
                    sample["state"] = state_batch[i]
            if meta_data:
                sample["meta_data"] = meta_data
            sample = self._output_transform(sample)

            transformed_actions.append(torch.from_numpy(sample["action"]))

        outputs["actions"] = torch.stack(transformed_actions, dim=0).to(
            outputs["actions"].device
        )
        outputs["actions"] = outputs["actions"][:, : self.num_action_chunks]

        return outputs

    def precision_processor(self, processed_obs):
        device = next(self.parameters()).device
        for key, value in processed_obs.items():
            if isinstance(value, list):
                processed_obs[key] = [
                    item.to(device=device).contiguous()
                    if torch.is_tensor(item)
                    else item
                    for item in value
                ]
            elif torch.is_tensor(value):
                processed_obs[key] = value.to(device=device).contiguous()
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if torch.is_tensor(sub_value):
                        processed_obs[key][sub_key] = sub_value.to(
                            device=device
                        ).contiguous()
        return processed_obs

    def forward(self, forward_type="default_forward", **kwargs):
        if "forward_inputs" in kwargs and "data" not in kwargs:
            kwargs["data"] = kwargs.pop("forward_inputs")

        if forward_type == "default_forward":
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError(f"Forward type {forward_type} not implemented")

    def default_forward(self, data, **kwargs):
        compute_values = kwargs.get("compute_values", False)
        chains = data["chains"]
        denoise_inds = data["denoise_inds"]
        if "tokenized_prompt" in data:
            observation = data
        else:
            observation = self.input_transform(data, transpose=False)

        device = chains.device

        raw_main_images = observation["observation/image"]
        raw_wrist_images = observation.get("observation/wrist_image", None)
        images, img_masks = self._process_images_for_training(
            raw_main_images, raw_wrist_images, device
        )

        target_dtype = next(self.parameters()).dtype
        lang_tokens = observation["tokenized_prompt"].to(device)
        lang_masks = observation["tokenized_prompt_mask"].to(device)

        state = observation["observation/state"].to(device=device)
        chains = data["chains"].to(device=device, dtype=target_dtype)

        log_probs, value_t, entropy = self.get_log_prob_value(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            chains,
            denoise_inds,
            compute_values,
        )
        log_probs = log_probs[
            :, :, : self.num_action_chunks, : self.config.action_env_dim
        ]
        entropy = entropy[:, :, : self.num_action_chunks, : self.config.action_env_dim]
        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs,
            "values": value_t,
            "entropy": entropy,
        }

    def _process_images_for_training(self, raw_main_images, raw_wrist_images, device):
        if torch.is_tensor(raw_main_images):
            raw_main_images = raw_main_images.cpu().numpy()
        if raw_wrist_images is not None and torch.is_tensor(raw_wrist_images):
            raw_wrist_images = raw_wrist_images.cpu().numpy()

        batch_size = raw_main_images.shape[0]
        base_pil_images = []
        for batch_idx in range(batch_size):
            img_np = raw_main_images[batch_idx]
            if img_np.dtype != np.uint8:
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                else:
                    img_np = img_np.astype(np.uint8)
            base_pil_images.append(Image.fromarray(img_np))
        wrist_pil_images = []
        if raw_wrist_images is not None:
            for batch_idx in range(batch_size):
                wrist_np = raw_wrist_images[batch_idx].astype(np.uint8)
                wrist_pil_images.append(Image.fromarray(wrist_np))
        images_list = []
        for batch_idx in range(batch_size):
            if wrist_pil_images:
                pil_pair = [base_pil_images[batch_idx], wrist_pil_images[batch_idx]]
                processed = self.process_images(pil_pair)  # [2, C, H, W]
            else:
                processed = self.process_images(
                    [base_pil_images[batch_idx]]
                )  # [1, C, H, W]
            images_list.append(processed)
        images = torch.stack(images_list, dim=0).to(
            device=device, dtype=next(self.parameters()).dtype
        )

        num_views = images.shape[1]
        required_num_images = 3
        if num_views < required_num_images:
            pad_size = required_num_images - num_views
            padding = torch.zeros(
                batch_size,
                pad_size,
                images.shape[2],
                images.shape[3],
                images.shape[4],
                dtype=images.dtype,
                device=device,
            )
            images = torch.cat([images, padding], dim=1)
        image_masks = torch.zeros(
            batch_size, required_num_images, dtype=torch.bool, device=device
        )
        image_masks[:, :num_views] = True

        return images, image_masks

    def _normalize_state(self, state):
        if not hasattr(self, "norm_stats") or self.norm_stats is None:
            return state
        if "state" not in self.norm_stats:
            return state
        stats = self.norm_stats["state"]
        mean = torch.tensor(stats["mean"], device=state.device, dtype=state.dtype)
        std = torch.tensor(stats["std"], device=state.device, dtype=state.dtype)
        normalized = (state - mean) / (std + 1e-6)
        return normalized

    def obs_processor(self, env_obs):
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        state = env_obs["states"]
        if torch.is_tensor(state):
            state = state.to(dtype=torch.float32)
        processed_obs["observation/state"] = state
        if "wrist_images" in env_obs:
            processed_obs["observation/wrist_image"] = env_obs["wrist_images"]
        return processed_obs

    @torch.no_grad()
    def sample_actions(
        self, processed_obs, noise=None, mode="train", compute_values=True
    ):
        original_training_mode = self.training
        self.eval()
        try:
            input_ids = processed_obs.get("tokenized_prompt")
            attention_mask = processed_obs.get("tokenized_prompt_mask")
            states = processed_obs["observation/state"].to(
                device=next(self.parameters()).device
            )

            raw_images = processed_obs["observation/image"]
            batch_size = raw_images.shape[0]
            device = states.device

            base_pil_images = []
            for batch_idx in range(batch_size):
                img_np = raw_images[batch_idx].cpu().numpy()
                if img_np.dtype != np.uint8:
                    if img_np.max() <= 1.0:
                        img_np = (img_np * 255).astype(np.uint8)
                    else:
                        img_np = img_np.astype(np.uint8)

                pil_img = Image.fromarray(img_np)
                base_pil_images.append(pil_img)

            wrist_pil_images = []
            if "observation/wrist_image" in processed_obs:
                wrist_raw = processed_obs["observation/wrist_image"]
                for batch_idx in range(batch_size):
                    wrist_np = wrist_raw[batch_idx].cpu().numpy().astype(np.uint8)
                    pil_img = Image.fromarray(wrist_np)
                    wrist_pil_images.append(pil_img)
            images_list = []
            for batch_idx in range(batch_size):
                if wrist_pil_images:
                    pil_pair = [base_pil_images[batch_idx], wrist_pil_images[batch_idx]]
                    processed = self.process_images(pil_pair)
                else:
                    processed = self.process_images([base_pil_images[batch_idx]])
                images_list.append(processed)
            images = torch.stack(images_list, dim=0).to(
                device=device, dtype=next(self.parameters()).dtype
            )
            num_views = images.shape[1]
            required_num_images = 3

            if num_views < required_num_images:
                pad_size = required_num_images - num_views
                padding = torch.zeros(
                    batch_size,
                    pad_size,
                    images.shape[2],
                    images.shape[3],
                    images.shape[4],
                    dtype=images.dtype,
                    device=device,
                )
                images = torch.cat([images, padding], dim=1)
            image_masks = torch.zeros(
                batch_size, required_num_images, dtype=torch.bool, device=device
            )
            image_masks[:, :num_views] = True
            target_dtype = next(self.parameters()).dtype
            num_steps = self.num_steps

            # Prefix forward: embed and cache LLM prefix
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                image_masks=image_masks,
            )
            prefix_att_2d_masks = make_attn_mask(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = make_attn_mask_4d(prefix_att_2d_masks)

            prefix_output, past_key_values = self._forward_mot_prefix(
                prefix_embeds=prefix_embs,
                mask=prefix_att_2d_masks_4d,
                position_embeddings=self.model.llm.rotary_emb(
                    prefix_embs, prefix_position_ids
                ),
                cache_position=prefix_position_ids,
            )

            # Init noise
            x_t = torch.randn(
                batch_size,
                self.config.chunk_size,
                self.config.action_dim,
                device=device,
                dtype=target_dtype,
            )

            chains = []
            log_probs = []
            values = []
            chains.append(x_t)

            if self.use_vlm_value:
                values_vlm = self.get_value_from_vlm(prefix_output)
            if self.config.joint_logprob:
                initial_log_prob = self.get_logprob_norm(
                    x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
                )
                log_probs.append(initial_log_prob)

            # Build denoise_inds
            if mode == "train":
                if self.config.joint_logprob:
                    denoise_inds = torch.arange(num_steps)
                else:
                    if getattr(self.config, "ignore_last", False):
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 2)] * num_steps
                        )
                    else:
                        denoise_inds = torch.tensor(
                            [random.randint(0, num_steps - 1)] * num_steps
                        )
            else:
                denoise_inds = torch.tensor([-1] * num_steps)
            denoise_inds = denoise_inds[None].repeat(batch_size, 1)

            # Diffusion loop
            for idx in range(num_steps):
                if idx == denoise_inds[0][idx]:
                    sample_mode = "train"
                else:
                    sample_mode = "eval"
                x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                    x_t,
                    idx,
                    states,
                    prefix_pad_masks,
                    past_key_values,
                    sample_mode,
                    num_steps,
                    compute_values,
                )
                x_t = x_t_mean + torch.randn_like(x_t) * x_t_std
                log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)
                values.append(value_t)
                chains.append(x_t)
                log_probs.append(log_prob)

            x_0 = x_t
            chains = torch.stack(chains, dim=1)

            # Post-process logprobs
            log_probs = torch.stack(log_probs, dim=1)[
                :, :, : self.num_action_chunks, : self.config.action_env_dim
            ]
            if self.config.joint_logprob:
                log_probs = log_probs.mean(dim=1)
            else:
                log_probs = log_probs[
                    torch.arange(log_probs.shape[0]),
                    denoise_inds[:, 0],
                ]

            # Post-process values
            if self.use_vlm_value:
                values = values_vlm[:, None]
            else:
                values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

            return {
                "actions": x_0,
                "chains": chains,
                "prev_logprobs": log_probs,
                "prev_values": values,
                "denoise_inds": denoise_inds,
            }
        finally:
            if original_training_mode:
                self.train()

    def get_logprob_norm(self, sample, mu, sigma):
        if self.config.safe_get_logprob:
            log_prob = -torch.pow((sample - mu), 2)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    def gaussian_entropy(self, sigma):
        import math

        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def _forward_mot_prefix(
        self,
        prefix_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> tuple[torch.Tensor, DynamicCache]:
        """Forward the LLM prefix only, building the KV cache.

        Uses each layer's forward() method so FSDP can properly gather/release
        parameters per layer.
        """
        hidden_states = prefix_embeds
        past_key_values = DynamicCache()
        # Cast mask to match model dtype (make_attn_mask_4d produces float32,
        # but SDPA requires mask dtype == query dtype).
        if mask is not None:
            mask = mask.to(dtype=hidden_states.dtype)

        for layer in self.model.llm.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=cache_position,
                past_key_value=past_key_values,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        prefix_output = self.model.llm.norm(hidden_states)
        return prefix_output, past_key_values

    def _forward_mot_suffix(
        self,
        suffix_embeds: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple] = None,
        past_key_values: Optional[DynamicCache] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Forward the action expert suffix only, attending to the LLM KV cache.

        Uses each layer's forward() method so FSDP can properly gather/release
        parameters per layer. Clones the KV cache to avoid mutating the
        original prefix cache (needed for multiple suffix calls).
        """
        hidden_states = suffix_embeds

        # Cast mask to match model dtype
        if mask is not None:
            mask = mask.to(dtype=hidden_states.dtype)

        # Shallow-clone the cache so cache.update() doesn't corrupt the original
        cloned_cache = DynamicCache()
        if past_key_values is not None:
            for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
                cloned_cache.key_cache.append(k)
                cloned_cache.value_cache.append(v)

        for layer in self.model.action_expert.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=cloned_cache,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        suffix_output = self.model.action_expert.norm(hidden_states)
        return suffix_output

    def get_suffix_out(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        batch_size = state.shape[0]
        device = state.device

        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=device)
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0).expand(batch_size)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
            states=state,
            noisy_actions=x_t,
            time=timestep,
        )
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask = prefix_pad_masks.unsqueeze(1).repeat(
            1, suffix_tokens.shape[1], 1
        )
        full_attn_mask = torch.cat([prefix_attn_mask, suffix_attn_mask], dim=-1)
        full_attn_mask = make_attn_mask_4d(full_attn_mask)
        full_positions = (
            prefix_pad_masks.sum(axis=-1).unsqueeze(-1)
            + torch.cumsum(suffix_mask, dim=-1)
            - 1
        )
        full_position_embeddings = self.model.llm.rotary_emb(
            suffix_tokens, full_positions
        )
        suffix_out = self._forward_mot_suffix(
            suffix_embeds=suffix_tokens,
            mask=full_attn_mask,
            position_embeddings=full_position_embeddings,
            past_key_values=past_key_values,
            position_ids=full_positions,
        )
        suffix_out = suffix_out.clone()[:, -self.config.chunk_size :]
        return suffix_out

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        bsize = state.shape[0]
        device = state.device
        if isinstance(idx, int):
            idx = torch.tensor(idx, device=device).expand(bsize)
        if self.config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(self.config.noise_level).to(device)
        # Timesteps: [1, 9/10, 8/10, ..., 1/10, 0] for 10 steps
        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        suffix_out = self.get_suffix_out(
            state,
            prefix_pad_masks,
            past_key_values,
            x_t,
            t_input,
        )
        v_t = self.model.action_out_proj(suffix_out)
        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            if self.config.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.config.chunk_size], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(
                suffix_out_value.to(self.value_head.weight.dtype)
            )[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device)
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.config.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(
                    suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
                )
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_log_prob_value(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        chains,
        denoise_inds,
        compute_values=False,
    ):
        bsize = state.shape[0]

        # When train_expert_only, the vision tower, LLM, and projector are all
        # frozen.  Wrapping the prefix computation in no_grad avoids storing
        # intermediate activations for these frozen modules, which is the main
        # source of memory savings.
        no_grad_ctx = (
            torch.no_grad()
            if getattr(self.config, "train_expert_only", False)
            else torch.enable_grad()
        )
        with no_grad_ctx:
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
                input_ids=lang_tokens,
                attention_mask=lang_masks,
                images=images,
                image_masks=img_masks,
            )
            prefix_att_2d_masks = make_attn_mask(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = make_attn_mask_4d(prefix_att_2d_masks)

            prefix_output, past_key_values = self._forward_mot_prefix(
                prefix_embeds=prefix_embs,
                mask=prefix_att_2d_masks_4d,
                position_embeddings=self.model.llm.rotary_emb(
                    prefix_embs, prefix_position_ids
                ),
                cache_position=prefix_position_ids,
            )

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            initial_entropy = self.gaussian_entropy(torch.ones_like(chains[:, 0]))
            chains_log_probs.append(initial_log_prob)
            chains_entropy.append(initial_entropy)
        else:
            num_steps = 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind].clone()
            chains_next = chains[torch.arange(bsize), denoise_ind + 1].clone()
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.config.num_steps,
                compute_values,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)

            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)

            if self.use_vlm_value:
                chains_values.append(self.get_value_from_vlm(prefix_output))
            else:
                chains_values.append(value_t)
        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)
        if self.config.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)

        return chains_log_probs, chains_values, chains_entropy

    def predict_action_batch(self, env_obs, **kwargs):
        mode = kwargs.get("mode", "train")
        compute_values = kwargs.get("compute_values", True)
        to_process_obs = self.obs_processor(env_obs)
        processed_obs = self.input_transform(to_process_obs, transpose=False)
        processed_obs = self.precision_processor(processed_obs)

        outputs = self.sample_actions(
            processed_obs=processed_obs, mode=mode, compute_values=compute_values
        )
        if hasattr(self, "_output_transform") and self._output_transform is not None:
            state_for_transform = processed_obs.get("observation/state")
            if state_for_transform is not None:
                if isinstance(state_for_transform, torch.Tensor):
                    state_numpy = state_for_transform.cpu().numpy()
                else:
                    state_numpy = state_for_transform
                meta_data = {"non_delta_mask": np.array(self.non_delta_mask)}
                outputs["state"] = state_numpy
                outputs["meta_data"] = meta_data
                outputs = self.output_transform(outputs)
            else:
                outputs = self.output_transform(outputs)
        else:
            pass

        actions = outputs["actions"][:, :, : self.config.action_env_dim]
        forward_inputs = {
            "chains": outputs["chains"],
            "denoise_inds": outputs["denoise_inds"],
        }
        if "tokenized_prompt" in processed_obs:
            forward_inputs["tokenized_prompt"] = processed_obs["tokenized_prompt"]
        if "tokenized_prompt_mask" in processed_obs:
            forward_inputs["tokenized_prompt_mask"] = processed_obs[
                "tokenized_prompt_mask"
            ]
        forward_inputs.update(to_process_obs)
        forward_inputs.pop("prompt", None)
        result = {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }

        return actions, result
