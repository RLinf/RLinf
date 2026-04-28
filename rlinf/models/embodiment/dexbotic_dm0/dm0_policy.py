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

import numpy as np
import torch
import torch.nn as nn
from dexbotic.data.dataset.transform.common import Pipeline, ToNumpy
from dexbotic.model.dm0.dm0_arch import (
    DM0ForCausalLM,
    make_attn_mask_2d,
    make_attn_mask_4d,
    make_suffix_attn_mask_2d,
)
from PIL import Image
from transformers import DynamicCache

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.utils.logging import get_logger


class DexboticDM0ForRLActionPrediction(BasePolicy, DM0ForCausalLM):
    _no_split_names = [
        "action_in_proj",
        "action_out_proj",
        "action_time_mlp_in",
        "action_time_mlp_out",
    ]

    def __init__(self, config):
        DM0ForCausalLM.__init__(self, config)
        # Use fine-grained FSDP wrapping (Qwen3MLP level), same approach as
        # dexbotic_pi_policy.py.  This avoids per-decoder-layer wrapping so
        # _merged_attention_forward can access sub-module parameters directly.
        self._no_split_modules = ["Qwen3MLP"]
        self.logger = get_logger()

        # Force uniform dtype so FSDP can flatten parameters without error.
        model_dtype = None
        if (
            hasattr(self.model, "llm")
            and hasattr(self.model.llm, "layers")
            and len(self.model.llm.layers) > 0
        ):
            for param in self.model.llm.layers[0].parameters():
                model_dtype = param.dtype
                break
        if model_dtype is None:
            all_params = list(self.model.parameters())
            if all_params:
                model_dtype = all_params[0].dtype
            else:
                model_dtype = torch.float32
        self.model = self.model.to(dtype=model_dtype)

        self.config = config
        self.num_steps = config.num_steps
        self.action_horizon = config.chunk_size
        self.num_action_chunks = getattr(
            config, "output_action_chunks", config.chunk_size
        )
        self.action_dim = config.action_dim
        self.non_delta_mask = getattr(config, "non_delta_mask", [6])
        self.global_step = 0
        self.use_vlm_value = False
        self.value_head = nn.Linear(config.action_config.hidden_size, 1)
        self.value_head = self.value_head.to(
            dtype=self.model.action_out_proj.weight.dtype
        )
        self._input_transform = None
        self._output_transform = None
        self.norm_stats = None
        self.dm0_tokenization = None

    def freeze_vlm(self):
        if not getattr(self.config, "train_expert_only", False):
            self.logger.warning("freeze_vlm() called but train_expert_only is False")
            return
        for component in ["mm_vision_tower", "llm", "mm_projector"]:
            mod = getattr(self.model, component, None)
            if mod is not None:
                mod.eval()
                for param in mod.parameters():
                    param.requires_grad = False

    def _read_normalization_stats(self, norm_stats_file):
        if not os.path.exists(norm_stats_file):
            raise FileNotFoundError(
                f"Normalization stats not found at {norm_stats_file}."
            )
        with open(norm_stats_file, "r") as f:
            norm_stats = json.load(f)
            if "norm_stats" in norm_stats:
                norm_stats = norm_stats["norm_stats"]
        return ToNumpy()(norm_stats)

    def setup_wrappers(self, transforms=(), output_transforms=()):
        self._input_transform = Pipeline(transforms) if transforms else None
        self._output_transform = (
            Pipeline(output_transforms) if output_transforms else None
        )

    def input_transform(self, obs: dict, transpose=True):
        if "prompt" in obs:
            prompts = obs["prompt"]
            if isinstance(prompts, str):
                prompts = [prompts]
            elif isinstance(prompts, torch.Tensor):
                prompts = [str(p) for p in prompts]
            batch_input_ids = []
            for prompt in prompts:
                tokenized = self.dm0_tokenization([{"from": "human", "value": prompt}])
                batch_input_ids.append(tokenized["input_ids"])

            batch_input_ids = torch.from_numpy(np.array(batch_input_ids))
            batch_attention_mask = batch_input_ids != self.tokenizer.pad_token_id

            obs["tokenized_prompt"] = batch_input_ids
            obs["tokenized_prompt_mask"] = batch_attention_mask

        if self._input_transform is not None and "observation/state" in obs:
            state_tensor = obs["observation/state"]
            if isinstance(state_tensor, torch.Tensor):
                state_value = state_tensor.cpu().float().numpy()
            else:
                state_value = state_tensor
            state_dict = self._input_transform({"state": state_value})
            obs["observation/state"] = state_dict["state"]
            obs["states"] = state_dict["state"]
        return obs

    def output_transform(self, outputs):
        if self._output_transform is None:
            self.logger.warning(
                "[output_transform] WARNING: _output_transform is None! "
                "Actions will NOT be denormalized!"
            )
            return outputs

        state_batch = outputs.get("state", None)
        meta_data = outputs.get("meta_data", {})
        batch_size = outputs["actions"].shape[0]
        transformed_actions = []

        for i in range(batch_size):
            sample = {"action": outputs["actions"][i].cpu().numpy()}
            if state_batch is not None:
                sample["state"] = (
                    state_batch[i].cpu().numpy()
                    if isinstance(state_batch, torch.Tensor)
                    else state_batch[i]
                )
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
        for i in range(batch_size):
            img_np = raw_main_images[i]
            if img_np.dtype != np.uint8:
                img_np = (
                    (img_np * 255).astype(np.uint8)
                    if img_np.max() <= 1.0
                    else img_np.astype(np.uint8)
                )
            base_pil_images.append(Image.fromarray(img_np))

        wrist_pil_images = []
        if raw_wrist_images is not None:
            for i in range(batch_size):
                wrist_np = raw_wrist_images[i].astype(np.uint8)
                wrist_pil_images.append(Image.fromarray(wrist_np))

        images_list = []
        for i in range(batch_size):
            pil_list = [base_pil_images[i]]
            if wrist_pil_images:
                pil_list.append(wrist_pil_images[i])
            images_list.append(self.process_images(pil_list))

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
                *images.shape[2:],
                dtype=images.dtype,
                device=device,
            )
            images = torch.cat([images, padding], dim=1)
        image_masks = torch.zeros(
            batch_size, required_num_images, dtype=torch.bool, device=device
        )
        image_masks[:, :num_views] = True
        return images, image_masks

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

    def _build_prefix_kv_cache(self, input_ids, attention_mask, images, image_masks):
        """Build KV cache from prefix (images + language) using per-layer LLM forward."""
        prefix_hidden_states, prefix_padding_mask, prefix_attn_mask = (
            self.get_prefix_hidden_states(
                input_ids, attention_mask, images, image_masks
            )
        )
        prefix_attn_mask_2d = make_attn_mask_2d(
            padding_mask=prefix_padding_mask, attn_mask=prefix_attn_mask
        )
        prefix_attn_mask_4d = make_attn_mask_4d(
            prefix_attn_mask_2d, dtype=prefix_hidden_states.dtype
        )
        positions = torch.cumsum(prefix_padding_mask, dim=1) - 1

        hidden_states = prefix_hidden_states
        past_key_values = DynamicCache()
        mask = prefix_attn_mask_4d.to(dtype=hidden_states.dtype)
        position_embeddings = self.model.llm.rotary_emb(hidden_states, positions)

        for layer in self.model.llm.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=positions,
                past_key_value=past_key_values,
                use_cache=True,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        del (
            hidden_states,
            mask,
            prefix_attn_mask_4d,
            prefix_attn_mask_2d,
            position_embeddings,
        )
        torch.cuda.empty_cache()
        return prefix_padding_mask, prefix_attn_mask, past_key_values

    def get_suffix_out(
        self,
        prefix_padding_mask,
        prefix_attn_mask,
        kv_cache,
        x_t,
        timestep,
    ):
        """Run suffix (action expert) using cached prefix KV, per-layer forward."""
        batch_size = x_t.shape[0]
        device = x_t.device

        model_dtype = self.model.action_in_proj.weight.dtype
        x_t = x_t.to(dtype=model_dtype)

        if not torch.is_tensor(timestep):
            timestep = torch.tensor(timestep, device=device)
        if timestep.dim() == 0:
            timestep = timestep.broadcast_to(batch_size)
        timestep = timestep.to(dtype=model_dtype)

        suffix_hidden_states, suffix_padding_mask, suffix_attn_mask = (
            self.get_suffix_hidden_states(x_t, timestep)
        )
        suffix_attn_mask_2d = make_suffix_attn_mask_2d(
            suffix_padding_mask=suffix_padding_mask,
            suffix_attn_mask=suffix_attn_mask,
            prefix_padding_mask=prefix_padding_mask,
            prefix_attn_mask=prefix_attn_mask,
        )
        full_attn_mask_4d = make_attn_mask_4d(
            suffix_attn_mask_2d, dtype=suffix_hidden_states.dtype
        )
        prefix_offsets = torch.sum(prefix_padding_mask, dim=-1)[:, None]
        full_positions = prefix_offsets + torch.cumsum(suffix_padding_mask, dim=1) - 1

        # Shallow-clone the KV cache so suffix forward doesn't corrupt the prefix cache
        cloned_cache = DynamicCache()
        for k, v in zip(kv_cache.key_cache, kv_cache.value_cache):
            cloned_cache.key_cache.append(k)
            cloned_cache.value_cache.append(v)

        hidden_states = suffix_hidden_states
        mask = full_attn_mask_4d.to(dtype=hidden_states.dtype)
        position_embeddings = self.model.llm.rotary_emb(hidden_states, full_positions)

        del full_attn_mask_4d, suffix_attn_mask_2d

        for layer in self.model.action_expert.model.layers:
            layer_outputs = layer(
                hidden_states,
                attention_mask=mask,
                position_ids=full_positions,
                past_key_value=cloned_cache,
                use_cache=False,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0]

        del cloned_cache, mask, position_embeddings
        hidden_states = self.model.action_expert.model.norm(hidden_states)
        suffix_out = hidden_states[:, -self.config.chunk_size :].clone()
        return suffix_out

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        prefix_padding_mask,
        prefix_attn_mask,
        kv_cache,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        bsize = x_t.shape[0]
        device = x_t.device
        if isinstance(idx, int):
            idx = torch.tensor(idx, device=device).expand(bsize)

        if self.config.noise_anneal:
            noise_start, noise_end, anneal_steps = self.config.noise_params
            noise_level = torch.tensor(
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps,
                device=device,
            )
        else:
            noise_level = torch.tensor(self.config.noise_level, device=device)

        timesteps = torch.linspace(1, 1 / denoise_steps, denoise_steps, device=device)
        timesteps = torch.cat([timesteps, torch.tensor([0.0], device=device)])
        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        suffix_out = self.get_suffix_out(
            prefix_padding_mask, prefix_attn_mask, kv_cache, x_t, t_input
        )
        v_t = self.model.action_out_proj(
            suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
        )

        if (
            self.config.add_value_head
            and compute_values
            and not self.config.value_after_vlm
        ):
            suffix_out_value = torch.mean(
                suffix_out[:, : self.config.chunk_size]
                if self.config.chunk_critic_input
                else suffix_out,
                dim=1,
                keepdim=False,
            )
            if self.config.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(
                suffix_out_value.to(self.value_head.weight.dtype)
            )[:, 0]
        else:
            value_t = torch.zeros(bsize, device=device)

        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x_t_mean = (1 - (t_input - delta)) * x0_pred + (t_input - delta) * x1_pred
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
                x_t_mean = (1 - (t_input - delta)) * x0_pred + (
                    t_input - delta - sigma_i**2 * delta / (2 * t_input)
                ) * x1_pred
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.config.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x_t_mean = (1 - (t_input - delta)) * x0_pred + (
                    t_input - delta
                ) * cos_term * x1_pred
                x_t_std = (t_input - delta) * sin_term
            elif self.config.noise_method == "flow_noise":
                x_t_mean = (1 - (t_input - delta)) * x0_pred + (
                    t_input - delta
                ) * x1_pred
                x_t_std = self.noise_head(
                    suffix_out.to(dtype=self.model.action_out_proj.weight.dtype)
                )
            else:
                raise ValueError(f"Invalid noise method: {self.config.noise_method}")
        else:
            raise ValueError(f"Invalid mode: {mode}")

        return x_t_mean, x_t_std, value_t

    def get_logprob_norm(self, sample, mu, sigma):
        if self.config.safe_get_logprob:
            return -torch.pow((sample - mu), 2)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * torch.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    def gaussian_entropy(self, sigma):
        import math

        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    @torch.no_grad()
    def sample_actions(
        self, processed_obs, noise=None, mode="train", compute_values=True
    ):
        input_ids = processed_obs.get("tokenized_prompt")
        attention_mask = processed_obs.get("tokenized_prompt_mask")
        states = processed_obs["observation/state"].to(
            device=next(self.parameters()).device
        )
        raw_images = processed_obs["observation/image"]
        batch_size = raw_images.shape[0]
        device = states.device

        base_pil_images = []
        for i in range(batch_size):
            img_np = raw_images[i].cpu().numpy()
            if img_np.dtype != np.uint8:
                img_np = (
                    (img_np * 255).astype(np.uint8)
                    if img_np.max() <= 1.0
                    else img_np.astype(np.uint8)
                )
            base_pil_images.append(Image.fromarray(img_np))

        wrist_pil_images = []
        if "observation/wrist_image" in processed_obs:
            for i in range(batch_size):
                wrist_np = (
                    processed_obs["observation/wrist_image"][i]
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                wrist_pil_images.append(Image.fromarray(wrist_np))

        images_list = []
        for i in range(batch_size):
            pil_list = [base_pil_images[i]]
            if wrist_pil_images:
                pil_list.append(wrist_pil_images[i])
            images_list.append(self.process_images(pil_list))

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
                *images.shape[2:],
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

        # Build prefix KV cache
        prefix_padding_mask, prefix_attn_mask, kv_cache = self._build_prefix_kv_cache(
            input_ids, attention_mask, images, image_masks
        )

        # Init noise
        x_t = torch.randn(
            batch_size,
            self.config.chunk_size,
            self.config.action_dim,
            device=device,
            dtype=target_dtype,
        )

        chains = [x_t]
        log_probs = []
        values = []

        if self.config.joint_logprob:
            log_probs.append(
                self.get_logprob_norm(x_t, torch.zeros_like(x_t), torch.ones_like(x_t))
            )

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
            sample_mode = "train" if idx == denoise_inds[0][idx] else "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                prefix_padding_mask,
                prefix_attn_mask,
                kv_cache,
                sample_mode,
                num_steps,
                compute_values,
            )
            x_t = x_t_mean + torch.randn_like(x_t) * x_t_std
            log_probs.append(self.get_logprob_norm(x_t, x_t_mean, x_t_std))
            values.append(value_t)
            chains.append(x_t)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)

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

        if self.use_vlm_value:
            raise NotImplementedError("use_vlm_value is not supported for DM0")
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

        return {
            "actions": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

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

        no_grad_ctx = (
            torch.no_grad()
            if getattr(self.config, "train_expert_only", False)
            else torch.enable_grad()
        )
        with no_grad_ctx:
            prefix_padding_mask, prefix_attn_mask, kv_cache = (
                self._build_prefix_kv_cache(lang_tokens, lang_masks, images, img_masks)
            )

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if self.config.joint_logprob:
            num_steps = self.config.num_steps
            chains_log_probs.append(
                self.get_logprob_norm(
                    chains[:, 0],
                    torch.zeros_like(chains[:, 0]),
                    torch.ones_like(chains[:, 0]),
                )
            )
            chains_entropy.append(self.gaussian_entropy(torch.ones_like(chains[:, 0])))
        else:
            num_steps = 1

        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[torch.arange(bsize), denoise_ind].clone()
            chains_next = chains[torch.arange(bsize), denoise_ind + 1].clone()
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                prefix_padding_mask,
                prefix_attn_mask,
                kv_cache,
                "train",
                self.config.num_steps,
                compute_values,
            )
            chains_log_probs.append(
                self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            )
            chains_entropy.append(self.gaussian_entropy(x_t_std))
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
        if self._output_transform is not None:
            state_for_transform = processed_obs.get("observation/state")
            if state_for_transform is not None:
                meta_data = {"non_delta_mask": np.array(self.non_delta_mask)}
                outputs["state"] = (
                    state_for_transform.cpu().numpy()
                    if isinstance(state_for_transform, torch.Tensor)
                    else state_for_transform
                )
                outputs["meta_data"] = meta_data
            outputs = self.output_transform(outputs)

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

        return actions, {
            "prev_logprobs": outputs["prev_logprobs"],
            "prev_values": outputs["prev_values"],
            "forward_inputs": forward_inputs,
        }
