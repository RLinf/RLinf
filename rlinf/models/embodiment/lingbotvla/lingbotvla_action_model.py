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
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from lingbotvla.data.vla_data.transform import (
    Normalizer,
    prepare_images,
    prepare_language,
    prepare_state,
)
from lingbotvla.models.module_utils import load_model_weights
from lingbotvla.models.vla.pi0.modeling_lingbot_vla import (
    LingbotVlaPolicy,
    make_att_2d_masks,
)
from transformers import AutoProcessor

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor


@dataclass
class Observation:
    image: Any
    state: Any
    prompt: Optional[Any] = None
    wrist_images: Optional[Any] = None

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            image=d.get("image"),
            state=d.get("state"),
            prompt=d.get("prompt"),
            wrist_images=d.get("wrist_images"),
        )


class LingbotvlaActionModel(nn.Module, BasePolicy):
    """
    LingbotVLA model wrapper for Reinforcement Learning (GRPO/PPO) & SFT.
    Strictly adapted to RLinf's OpenPI ecosystem without code simplification.
    """

    @property
    def _no_split_modules(self) -> list[str]:
        no_split_modules = [
            "Qwen2DecoderLayer",
            "Qwen2_5_VLDecoderLayer",
            "Qwen2_5_VLVisionBlock",
        ]
        if getattr(self.config, "noise_method", "flow_sde") == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "visual",
            "embed_tokens",
            "action_in_proj",
            "action_out_proj",
            "state_proj",
            "lm_head",
            "depth_align_embs",
            "value_head",
        ]

    def __init__(self, config, torch_dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.torch_dtype = torch_dtype
        self.logger = get_logger()
        self.global_step = 0

        self.action_dim = getattr(config, "action_dim", 75)
        self.action_chunk = getattr(
            config, "action_chunk", getattr(config, "num_action_chunks", 50)
        )
        self.action_env_dim = getattr(config, "action_env_dim", self.action_dim)
        self.num_steps = getattr(config, "num_steps", 10)
        self.noise_method = getattr(config, "noise_method", "flow_sde")

        assert not (
            getattr(self.config, "double_layer", False)
            and getattr(self.config, "joint_logprob", False)
        ), "double_layer and joint_logprob can not be set at the same time"

        lingbotvla_cfg = getattr(
            config, "lingbotvla", getattr(config, "lingbot", config)
        )
        config_path = getattr(
            lingbotvla_cfg,
            "config_path",
            os.path.join(os.environ.get("LINGBOT_VLA_PATH", ""), "lingbot-vla-4b"),
        )
        from lerobot.configs.policies import PreTrainedConfig

        qwen_config = PreTrainedConfig.from_pretrained(config_path)

        qwen_config.train_state_proj = True
        qwen_config.adanorm_time = True
        qwen_config.split_gate_liner = False
        qwen_config.nosplit_gate_liner = False
        qwen_config.separate_time_proj = False
        qwen_config.old_adanorm = True
        qwen_config.final_norm_adanorm = False
        qwen_config.freeze_vision_encoder = True
        qwen_config.tokenizer_max_length = 24
        qwen_config.attention_implementation = "flex"
        qwen_config.enable_expert_vision = False
        qwen_config.expert_vision_type = None
        qwen_config.action_dim = self.action_dim
        qwen_config.max_action_dim = getattr(lingbotvla_cfg, "max_action_dim", 75)
        qwen_config.max_state_dim = getattr(lingbotvla_cfg, "max_state_dim", 75)
        qwen_config.n_action_steps = self.action_chunk
        qwen_config.vlm_repo_id = None
        qwen_config.expert_vision_path = None
        qwen_config.tokenizer_path = config.tokenizer_path
        qwen_config.loss_type = "L1_fm"
        qwen_config.align_params = {}
        qwen_config.norm_qkv = False
        qwen_config.use_lm_head = False
        qwen_config.vocab_size = 151936

        self.vla_model = LingbotVlaPolicy(
            config=qwen_config, tokenizer_path=config.tokenizer_path
        ).to(self.torch_dtype)

        if getattr(config, "model_path", None):
            load_model_weights(
                self.vla_model,
                config.model_path,
                init_device="cuda",
                post_training=True,
                adanorm_time=True,
            )

        self.use_vlm_value = getattr(self.config, "value_after_vlm", False) and getattr(
            self.config, "add_value_head", False
        )
        if getattr(self.config, "add_value_head", False):
            proj_width = self.vla_model.model.config.proj_width
            value_head_hidden_sizes = (
                (1024, 512, 256)
                if "pi05" in getattr(self.config, "config_name", "")
                else (512, 256, 128)
            )
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=value_head_hidden_sizes,
                output_dim=1,
                activation="relu",
                bias_last=True,
            ).to(self.torch_dtype)

        # noise head for flow-noise
        if self.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=self.vla_model.model.config.proj_width,
                out_dim=self.action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=getattr(
                    self.config, "noise_logvar_range", [0.08, 0.16]
                ),
                noise_scheduler_type="learn",
            ).to(self.torch_dtype)

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

        self.processor = AutoProcessor.from_pretrained(config.tokenizer_path)
        self.language_tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor

        stats_json_path = getattr(
            lingbotvla_cfg,
            "stats_path",
            os.path.join(
                os.environ.get("LINGBOT_VLA_PATH", ""),
                "assets/norm_stats/robotwin_50.json",
            ),
        )
        with open(stats_json_path, "r") as f:
            raw_stats = json.load(f)

        self.norm_stats = raw_stats.get("norm_stats", raw_stats.get("stats", raw_stats))
        self.normalizer = Normalizer(
            norm_stats=self.norm_stats,
            from_file=True,
            data_type="robotwin",
            norm_type={
                "observation.images.cam_high": "identity",
                "observation.images.cam_left_wrist": "identity",
                "observation.images.cam_right_wrist": "identity",
                "observation.state": "bounds_99_woclip",
                "action": "bounds_99_woclip",
            },
        )

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.vla_model, "gradient_checkpointing_enable"):
            self.vla_model.gradient_checkpointing_enable(**kwargs)

    def set_global_step(self, global_step):
        self.global_step = global_step

    def obs_processor(self, env_obs):
        processed_obs = {
            "image": env_obs.get("main_images", env_obs.get("prep_images")),
            "prompt": env_obs.get("task_descriptions", env_obs.get("prompt")),
        }
        if "calvin" in getattr(self.config, "config_name", ""):
            state = env_obs["states"]
            processed_obs["observation/state_ee_pos"] = state[:, :3]
            processed_obs["observation/state_ee_rot"] = state[:, 3:6]
            processed_obs["observation/state_gripper"] = state[:, 6:7]
            processed_obs["state"] = state
        else:
            processed_obs["state"] = env_obs.get("states", env_obs.get("prep_state"))

        if "wrist_images" in env_obs and env_obs["wrist_images"] is not None:
            processed_obs["wrist_images"] = env_obs["wrist_images"]

        return processed_obs

    def _preprocess_observation(self, observation: Observation, train: bool = False):
        device = next(self.parameters()).device

        imgs = observation.image
        states_raw = observation.state

        batch_size = (
            states_raw.shape[0] if hasattr(states_raw, "shape") else len(states_raw)
        )
        instruction = (
            observation.prompt if observation.prompt is not None else [""] * batch_size
        )
        wrist_images = getattr(observation, "wrist_images", [])

        if isinstance(states_raw, torch.Tensor):
            states_tensor = states_raw.detach().cpu().to(dtype=torch.float32)
        else:
            states_tensor = torch.from_numpy(np.stack(states_raw)).to(
                dtype=torch.float32
            )

        prep_images_list, prep_img_masks_list = [], []
        lang_tokens_list, lang_masks_list, prep_state_list = [], [], []

        for i in range(batch_size):
            curr_img = imgs[i]
            if isinstance(curr_img, torch.Tensor):
                curr_img = curr_img.detach().cpu()
            elif isinstance(curr_img, np.ndarray):
                curr_img = torch.from_numpy(curr_img)

            if curr_img.ndim == 3 and curr_img.shape[-1] in [1, 3]:
                curr_img = curr_img.permute(2, 0, 1)

            if curr_img.is_floating_point():
                curr_img = (
                    (curr_img * 255).to(torch.uint8)
                    if curr_img.max() <= 1.0
                    else curr_img.to(torch.uint8)
                )
            else:
                curr_img = curr_img.to(torch.uint8)

            curr_left = curr_img
            curr_right = curr_img
            if (
                wrist_images is not None
                and len(wrist_images) > i
                and wrist_images[i] is not None
            ):
                if len(wrist_images[i]) > 0 and wrist_images[i][0] is not None:
                    curr_left = wrist_images[i][0]
                    if isinstance(curr_left, torch.Tensor):
                        curr_left = curr_left.detach().cpu()
                    elif isinstance(curr_left, np.ndarray):
                        curr_left = torch.from_numpy(curr_left)

                    if curr_left.ndim == 3 and curr_left.shape[-1] in [1, 3]:
                        curr_left = curr_left.permute(2, 0, 1)
                    curr_left = (
                        (curr_left * 255).to(torch.uint8)
                        if curr_left.is_floating_point() and curr_left.max() <= 1.0
                        else curr_left.to(torch.uint8)
                    )

                if len(wrist_images[i]) > 1 and wrist_images[i][1] is not None:
                    curr_right = wrist_images[i][1]
                    if isinstance(curr_right, torch.Tensor):
                        curr_right = curr_right.detach().cpu()
                    elif isinstance(curr_right, np.ndarray):
                        curr_right = torch.from_numpy(curr_right)

                    if curr_right.ndim == 3 and curr_right.shape[-1] in [1, 3]:
                        curr_right = curr_right.permute(2, 0, 1)
                    curr_right = (
                        (curr_right * 255).to(torch.uint8)
                        if curr_right.is_floating_point() and curr_right.max() <= 1.0
                        else curr_right.to(torch.uint8)
                    )

            norm_obs = self.normalizer.normalize(
                {
                    "observation.state": states_tensor[i],
                }
            )

            processor_obs = {
                "image": {
                    "base_0_rgb": curr_img,
                    "left_wrist_0_rgb": curr_left,
                    "right_wrist_0_rgb": curr_right,
                },
                "state": norm_obs["observation.state"].to(torch.float32),
                "prompt": [instruction[i]],
            }

            prep_state = prepare_state(
                self.vla_model.model.config, processor_obs
            ).unsqueeze(0)
            lang_tokens, lang_masks = prepare_language(
                self.vla_model.model.config, self.language_tokenizer, processor_obs
            )

            prep_images, prep_img_masks, _ = prepare_images(
                self.vla_model.model.config, self.image_processor, processor_obs
            )

            prep_images_list.append(prep_images)
            prep_img_masks_list.append(prep_img_masks)
            lang_tokens_list.append(lang_tokens.unsqueeze(0))
            lang_masks_list.append(lang_masks.unsqueeze(0))
            prep_state_list.append(prep_state)

        batched_images = torch.stack(prep_images_list, dim=0).to(
            device, dtype=self.torch_dtype
        )
        batched_img_masks = torch.stack(prep_img_masks_list, dim=0).to(device)
        lang_tokens = torch.cat(lang_tokens_list, dim=0).to(device)
        lang_masks = torch.cat(lang_masks_list, dim=0).to(device)
        state = torch.cat(prep_state_list, dim=0).to(device, dtype=self.torch_dtype)

        return [batched_images], [batched_img_masks], lang_tokens, lang_masks, state

    def output_transform(self, outputs: dict) -> dict:
        action_pred = (
            outputs["actions"][:, : self.action_chunk, :]
            .to(torch.float32)
            .cpu()
            .numpy()
        )
        unnorm_data = self.normalizer.unnormalize({"action": action_pred})
        outputs["actions"] = torch.from_numpy(unnorm_data["action"])
        return outputs

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        processed_obs = self.obs_processor(env_obs)
        observation = Observation.from_dict(processed_obs)

        is_dsrl_train = getattr(self.config, "use_dsrl", False) and mode == "train"
        if is_dsrl_train:
            dsrl_obs = {"images": [env_obs["main_images"]], "states": env_obs["states"]}
            noise_actions, noise_logprob, _ = self.sac_forward(
                dsrl_obs, train=False, mode=mode
            )
            outputs = self.sample_actions(
                observation,
                noise=noise_actions,
                mode="eval",
                compute_values=compute_values,
            )
            real_actions = self.output_transform(
                {"actions": outputs["actions"], "state": observation.state}
            )["actions"].numpy()

            actions = real_actions
            prev_logprobs = noise_logprob
            prev_values = outputs.get("prev_values")
            forward_action = noise_actions
        else:
            outputs = self.sample_actions(
                observation, mode=mode, compute_values=compute_values
            )
            actions = self.output_transform(
                {"actions": outputs["actions"], "state": observation.state}
            )["actions"].numpy()

            prev_logprobs = outputs["prev_logprobs"]
            prev_values = outputs["prev_values"]
            forward_action = None

        forward_inputs = {
            "chains": outputs["chains"].cpu(),
            "denoise_inds": outputs["denoise_inds"].cpu(),
            "lang_tokens": outputs["lang_tokens"].cpu(),
            "lang_masks": outputs["lang_masks"].cpu(),
        }
        if forward_action is not None:
            forward_inputs["action"] = forward_action

        cloned_obs = copy_dict_tensor(
            {
                k: v
                for k, v in env_obs.items()
                if k not in ["task_descriptions", "prompt"] and v is not None
            }
        )
        forward_inputs.update(cloned_obs)

        result = {
            "prev_logprobs": prev_logprobs.to(torch.float32)
            if torch.is_tensor(prev_logprobs)
            else prev_logprobs,
            "prev_values": prev_values.to(torch.float32)
            if torch.is_tensor(prev_values)
            else prev_values,
            "forward_inputs": forward_inputs,
        }
        return actions, result

    def sample_noise(self, shape, device):
        return torch.randn(shape, device=device)

    @torch.no_grad()
    def sample_actions(
        self,
        observation: Observation,
        noise=None,
        mode="train",
        compute_values=True,
    ) -> dict[str, Any]:
        """Do a full inference forward and compute the action"""
        bsize = (
            observation.state.shape[0]
            if hasattr(observation.state, "shape")
            else len(observation.state)
        )
        device = next(self.parameters()).device
        num_steps = self.config.num_steps
        max_act_dim = self.vla_model.model.config.max_action_dim
        if noise is None:
            actions_shape = (
                bsize,
                getattr(self.config, "action_horizon", self.action_chunk),
                max_act_dim,
            )
            noise = self.sample_noise(actions_shape, device)
        else:
            noise = noise.to(self.torch_dtype)
            if noise.shape[-1] < max_act_dim:
                pad_noise = self.sample_noise(
                    (*noise.shape[:-1], max_act_dim - noise.shape[-1]), device
                )
                noise = torch.cat([noise, pad_noise], dim=-1)

        images, img_masks, lang_tokens, lang_masks, state = (
            self._preprocess_observation(observation, train=False)
        )

        vla_images = images[0] if isinstance(images, list) else images
        vla_img_masks = img_masks[0] if isinstance(img_masks, list) else img_masks

        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.vla_model.model.embed_prefix(
                vla_images, vla_img_masks, lang_tokens, lang_masks, False
            )
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        outputs, past_key_values = self.vla_model.model.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        prefix_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        x_t = noise
        chains = []
        log_probs = []
        values = []
        chains.append(x_t)

        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(prefix_output, prefix_pad_masks)

        if getattr(self.config, "joint_logprob", False):
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(noise), torch.ones_like(noise)
            )
            log_probs.append(initial_log_prob)

        if mode == "train":
            if getattr(self.config, "joint_logprob", False):
                denoise_inds = torch.arange(num_steps, device=device)
            else:
                if getattr(self.config, "ignore_last", False):
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 2)] * num_steps, device=device
                    )
                else:
                    denoise_inds = torch.tensor(
                        [random.randint(0, num_steps - 1)] * num_steps, device=device
                    )
        else:
            denoise_inds = torch.tensor([-1] * num_steps, device=device)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        for idx in range(num_steps):
            if idx == denoise_inds[0][idx]:
                sample_mode = "train"
            else:
                sample_mode = "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                num_steps,
                compute_values,
            )
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)

        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.action_chunk, : self.action_env_dim
        ]
        if getattr(self.config, "joint_logprob", False):
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[
                torch.arange(log_probs.shape[0]),
                denoise_inds[:, 0],
            ]

        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

        return {
            "actions": x_0[:, :, : self.action_dim],
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
        }

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
            idx = torch.tensor(idx).expand(bsize)

        if getattr(self.config, "noise_anneal", False):
            noise_start, noise_end, anneal_steps = getattr(
                self.config, "noise_params", [0.7, 0.3, 400]
            )
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(self.global_step, anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(getattr(self.config, "noise_level", 0.5)).to(
                device
            )

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
        v_t = self.vla_model.model.action_out_proj(suffix_out)

        if (
            getattr(self.config, "add_value_head", False)
            and compute_values
            and not getattr(self.config, "value_after_vlm", False)
        ):
            if getattr(self.config, "chunk_critic_input", False):
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            if getattr(self.config, "detach_critic_input", False):
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device, dtype=self.torch_dtype)

        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.noise_method == "flow_sde":
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
            elif self.noise_method == "flow_cps":
                pi = torch.pi
                cos_term = torch.cos(pi * noise_level / 2).to(device)
                sin_term = torch.sin(pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out)
            else:
                raise ValueError(f"Invalid noise method: {self.noise_method}")
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    def get_suffix_out(
        self, state, prefix_pad_masks, past_key_values, x_t, timestep, expert_imgs=None
    ):
        x_t_padded = x_t.to(self.torch_dtype)

        time_embs, suffix_embs, suffix_pad_masks, suffix_att_masks = (
            self.vla_model.model.embed_suffix(state, x_t_padded, timestep, expert_imgs)
        )

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vla_model.model.qwenvl_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
            ada_cond=time_embs
            if getattr(self.vla_model.model.config, "adanorm_time", False)
            else None,
        )
        suffix_out = outputs_embeds[1]

        return suffix_out[:, -self.action_chunk :]

    def get_logprob_norm(self, sample, mu, sigma):
        sample = sample.to(torch.float32)
        mu = mu.to(torch.float32)
        sigma = sigma.to(torch.float32)

        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)

        if getattr(self.config, "safe_get_logprob", False):
            log_prob = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        else:
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)

        return log_prob

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
        vla_images = images[0] if isinstance(images, list) else images
        vla_img_masks = img_masks[0] if isinstance(img_masks, list) else img_masks

        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.vla_model.model.embed_prefix(
                vla_images, vla_img_masks, lang_tokens, lang_masks, False
            )
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        outputs, past_key_values = self.vla_model.model.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        prefix_output = outputs[0] if isinstance(outputs, (list, tuple)) else outputs

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if getattr(self.config, "joint_logprob", False):
            num_steps = self.num_steps
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
            chains_pre = chains[torch.arange(bsize), denoise_ind]
            chains_next = chains[torch.arange(bsize), denoise_ind + 1]

            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                state,
                prefix_pad_masks,
                past_key_values,
                "train",
                self.num_steps,
                compute_values,
            )

            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            entropy = self.gaussian_entropy(x_t_std)
            chains_log_probs.append(log_probs)
            chains_entropy.append(entropy)

            if not getattr(self.config, "value_after_vlm", False):
                chains_values.append(value_t)

        if getattr(self.config, "value_after_vlm", False):
            if getattr(self.config, "add_value_head", False):
                chains_values.append(
                    self.get_value_from_vlm(prefix_output, prefix_pad_masks)
                )
            else:
                chains_values.append(
                    torch.zeros(bsize, device=state.device, dtype=self.torch_dtype)
                )

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        if self.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)

        return chains_log_probs, chains_values, chains_entropy

    def get_value_from_vlm(self, prefix_output, prefix_pad_masks):
        mask = prefix_pad_masks.unsqueeze(-1).to(prefix_output.dtype)
        sum_hidden = (prefix_output * mask).sum(dim=1)
        valid_token_count = mask.sum(dim=1).clamp(min=1e-6)
        prefix_out_value = sum_hidden / valid_token_count
        values_vlm = self.value_head(prefix_out_value.to(self.torch_dtype)).squeeze(-1)
        return values_vlm

    def gaussian_entropy(self, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * (sigma_safe**2))
        return entropy

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sft_forward(self, data, **kwargs):
        dtype = self.torch_dtype
        images = data["images"].to(dtype)
        state = data["state"].to(dtype)
        actions = data["actions"].to(dtype)

        total_loss, loss_vla, loss_depth, loss_dict, depth_preds = self.vla_model(
            images=images,
            img_masks=data["img_masks"],
            state=state,
            lang_tokens=data["lang_tokens"],
            lang_masks=data["lang_masks"],
            actions=actions,
            use_ki=False,
            norm_qkv=self.vla_model.model.config.norm_qkv,
        )
        return {"loss": total_loss, "l1_loss": loss_vla, **loss_dict}

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        compute_values = kwargs.get("compute_values", False)
        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]

        obs_dict = {
            "image": forward_inputs.get(
                "main_images",
                forward_inputs.get("prep_images", forward_inputs.get("images")),
            ),
            "state": forward_inputs.get("states", forward_inputs.get("prep_state")),
        }
        if "wrist_images" in forward_inputs:
            obs_dict["wrist_images"] = forward_inputs["wrist_images"]

        observation = Observation.from_dict(obs_dict)

        images, img_masks, _, _, state = self._preprocess_observation(
            observation, train=False
        )

        lang_tokens = forward_inputs["lang_tokens"]
        lang_masks = forward_inputs["lang_masks"]

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

        log_probs = log_probs[:, :, : self.action_chunk, : self.action_env_dim]
        entropy = entropy[:, :, : self.action_chunk, : self.action_env_dim]

        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs.to(torch.float32),
            "values": value_t.to(torch.float32),
            "entropy": entropy.to(torch.float32),
        }
