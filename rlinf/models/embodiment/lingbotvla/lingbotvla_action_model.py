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
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from lingbotvla.data.vla_data.transform import (
    Normalizer,
    prepare_images,
    prepare_language,
    prepare_state,
)
from lingbotvla.models import build_foundation_model, build_processor
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class LingbotVLAForRLActionPrediction(nn.Module, BasePolicy):
    """LingBot VLA model wrapper for Reinforcement Learning action prediction.

    This class adapts the LingBot VLA flow-matching model to output actions
    and log-probabilities compatible with RL algorithms like GRPO and PPO
    using ODE-SDE mixed sampling.
    """

    def __init__(self, cfg, torch_dtype=torch.bfloat16):
        """Initializes the LingBot VLA model for RL.

        Args:
            cfg: Configuration object containing model parameters.
            torch_dtype: The torch data type for model weights (default: torch.bfloat16).
        """
        super().__init__()
        self.config = cfg
        self.action_dim = cfg.action_dim
        self.num_action_chunks = getattr(cfg, "num_action_chunks", 50)
        self.torch_dtype = torch_dtype

        lingbot_cfg = getattr(cfg, "lingbot", cfg)
        self.num_steps = getattr(cfg, "num_steps", 10)
        self.noise_method = getattr(cfg, "noise_method", "flow_sde")
        self.noise_level = getattr(cfg, "noise_level", 0.5)
        self.joint_logprob = getattr(cfg, "joint_logprob", False)
        self.action_env_dim = getattr(cfg, "action_env_dim", self.action_dim)
        self.global_step = 0

        # --- 1. Load LingBot VLA Foundation Model ---
        config_kwargs = {
            "vlm_repo_id": None,
            "action_dim": self.action_dim,
            "max_action_dim": 75,
            "max_state_dim": 75,
            "chunk_size": self.num_action_chunks,
            "tokenizer_path": cfg.tokenizer_path,
            "post_training": True,
            "incremental_training": False,
            "depth_incremental_training": False,
            "norm_qkv": False,
            "enable_expert_vision": False,
            "expert_vision_type": None,
            "expert_vision_path": None,
            "adanorm_time": True,
            "split_gate_liner": False,
            "nosplit_gate_liner": False,
            "separate_time_proj": False,
            "old_adanorm": True,
            "final_norm_adanorm": False,
            "loss_type": "L1_fm",
            "align_params": {},
        }

        self.vla_model = build_foundation_model(
            config_path=getattr(
                lingbot_cfg,
                "config_path",
                os.path.join(os.environ.get("LINGBOT_VLA_PATH", ""), "lingbot-vla-4b"),
            ),
            weights_path=cfg.model_path,
            torch_dtype=str(torch_dtype).split(".")[-1],
            init_device="cuda",
            freeze_vision_encoder=False,
            tokenizer_max_length=24,
            vocab_size=151936,
            use_lm_head=False,
            force_use_huggingface=False,
            config_kwargs=config_kwargs,
        )
        self.vla_model.eval()

        # --- 2. Load Processors ---
        self.processor = build_processor(cfg.tokenizer_path)
        self.language_tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor

        # --- 3. Build Normalizer ---
        stats_json_path = getattr(
            lingbot_cfg,
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

    @property
    def _no_split_modules(self) -> list[str]:
        return ["Qwen2DecoderLayer", "Qwen2_5_VLDecoderLayer", "Qwen2_5_VLVisionBlock"]

    @property
    def _no_split_names(self) -> list[str]:
        return []

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.vla_model, "gradient_checkpointing_enable"):
            self.vla_model.gradient_checkpointing_enable(**kwargs)

    def set_global_step(self, global_step):
        self.global_step = global_step

    def get_logprob_norm(self, sample, mu, sigma):
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
            2 * math.pi * torch.ones_like(sample)
        )
        exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
        log_prob = constant_term + exponent_term
        log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob.to(self.torch_dtype)

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        state,
        prefix_pad_masks,
        past_key_values,
        mode,
        denoise_steps,
        expert_imgs=None,
    ):
        bsize = state.shape[0]
        device = state.device
        dtype = self.torch_dtype

        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)

        noise_level = torch.tensor(self.noise_level, device=device, dtype=dtype)
        timesteps = torch.linspace(
            1, 1 / denoise_steps, denoise_steps, device=device, dtype=dtype
        )
        timesteps = torch.cat(
            [timesteps, torch.tensor([0.0], device=device, dtype=dtype)]
        )

        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        x_t = x_t.to(dtype)
        t_input = t_input.to(dtype)

        v_t = self.vla_model.model.predict_velocity(
            state, prefix_pad_masks, past_key_values, x_t, expert_imgs, timestep=t_input
        )
        v_t = v_t.to(dtype)

        delta_exp = delta[:, None, None].expand_as(x_t).to(dtype)
        t_input_exp = t_input[:, None, None].expand_as(x_t).to(dtype)

        x0_pred = x_t - v_t * t_input_exp
        x1_pred = x_t + v_t * (1 - t_input_exp)

        if mode == "eval":
            x0_weight = (1 - (t_input_exp - delta_exp)).to(dtype)
            x1_weight = (t_input_exp - delta_exp).to(dtype)
            x_t_std = torch.zeros_like(t_input_exp, dtype=dtype)
        else:
            sigmas = (
                noise_level
                * torch.sqrt(
                    timesteps
                    / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                )[:-1]
            )
            sigma_i = sigmas[idx][:, None, None].expand_as(x_t).to(dtype)
            x0_weight = (torch.ones_like(t_input_exp) - (t_input_exp - delta_exp)).to(
                dtype
            )
            x1_weight = (
                t_input_exp - delta_exp - sigma_i**2 * delta_exp / (2 * t_input_exp)
            ).to(dtype)
            x_t_std = (torch.sqrt(delta_exp) * sigma_i).to(dtype)

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return (
            x_t_mean.to(dtype),
            x_t_std.to(dtype),
            torch.zeros((bsize), device=device, dtype=dtype),
        )

    @torch.no_grad()
    def sample_actions_rl(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        expert_imgs=None,
        vlm_causal=False,
        mode="train",
    ):
        bsize = state.shape[0]
        device = state.device
        dtype = self.torch_dtype
        num_steps = self.num_steps

        actions_shape = (
            bsize,
            self.num_action_chunks,
            self.vla_model.model.config.max_action_dim,
        )
        x_t = torch.randn(actions_shape, device=device, dtype=dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks = (
            self.vla_model.model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, vlm_causal
            )
        )
        from lingbotvla.models.vla.pi0.modeling_lingbot_vla import make_att_2d_masks

        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.vla_model.model.qwenvl_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        chains = [x_t]
        log_probs = []

        if mode == "train":
            denoise_inds = torch.tensor([random.randint(0, num_steps - 1)] * num_steps)
        else:
            denoise_inds = torch.tensor([-1] * num_steps)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        for idx in range(num_steps):
            sample_mode = (
                "train" if (mode == "train" and idx == denoise_inds[0][idx]) else "eval"
            )
            x_t_mean, x_t_std, _ = self.sample_mean_var_val(
                x_t,
                idx,
                state,
                prefix_pad_masks,
                past_key_values,
                sample_mode,
                num_steps,
                expert_imgs,
            )
            x_t = x_t_mean + torch.randn_like(x_t_mean, dtype=dtype) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            chains.append(x_t)
            log_probs.append(log_prob)

        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.num_action_chunks, : self.action_env_dim
        ]

        log_probs = log_probs[torch.arange(log_probs.shape[0]), denoise_inds[:, 0]]

        return x_t, chains, log_probs, denoise_inds

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "train",
        compute_values: bool = True,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Predicts a batch of actions from environment observations and computes logprobs.

        Args:
            env_obs: Dictionary containing environment observations (images, states, text).
            mode: Operating mode, either "train" (uses SDE) or "eval" (uses ODE).
            compute_values: Whether to compute value baseline.
            **kwargs: Additional arguments.

        Returns:
            A tuple containing the predicted actions tensor and a result dictionary
            with logprobs and preprocessed forward inputs.
        """
        batch_size = len(env_obs["task_descriptions"])
        device = next(self.parameters()).device

        actions_list = []
        rl_chains_list = []
        rl_logprobs_list = []
        rl_denoise_inds_list = []

        prep_images_list = []
        prep_img_masks_list = []
        lang_tokens_list = []
        lang_masks_list = []
        prep_state_list = []

        def process_img(
            img: Union[torch.Tensor, np.ndarray, None],
        ) -> Optional[np.ndarray]:
            if img is None:
                return None
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu().numpy()
            if img.dtype == np.float32 or img.dtype == np.float64:
                if img.max() <= 1.0:
                    img = (img * 255.0).clip(0, 255)
            img = img.astype(np.uint8)
            cam = Image.fromarray(img).resize((224, 224), Image.BILINEAR)
            cam = np.transpose(np.array(cam), (2, 0, 1)) / 255.0
            return cam

        for i in range(batch_size):
            instruction = env_obs["task_descriptions"][i]
            cam_high = process_img(env_obs["main_images"][i])

            if (
                env_obs.get("wrist_images") is not None
                and len(env_obs["wrist_images"]) > i
                and env_obs["wrist_images"][i] is not None
                and len(env_obs["wrist_images"][i]) > 0
            ):
                cam_left = process_img(env_obs["wrist_images"][i][0])
                if len(env_obs["wrist_images"][i]) > 1:
                    cam_right = process_img(env_obs["wrist_images"][i][1])
                else:
                    cam_right = cam_left
            else:
                cam_left = cam_right = cam_high

            state = (
                env_obs["states"][i]
                if "states" in env_obs
                else np.zeros(14, dtype=np.float32)
            )
            if isinstance(state, torch.Tensor):
                state = state.detach().cpu().numpy()

            obs_dict_raw = {
                "observation.images.cam_high": torch.from_numpy(cam_high).float(),
                "observation.images.cam_left_wrist": torch.from_numpy(cam_left).float(),
                "observation.images.cam_right_wrist": torch.from_numpy(
                    cam_right
                ).float(),
                "observation.state": torch.from_numpy(state).float(),
                "task": instruction,
            }

            norm_obs = self.normalizer.normalize(obs_dict_raw)

            base_image = (norm_obs["observation.images.cam_high"] * 255).to(torch.uint8)
            left_wrist_image = (norm_obs["observation.images.cam_left_wrist"] * 255).to(
                torch.uint8
            )
            right_wrist_image = (
                norm_obs["observation.images.cam_right_wrist"] * 255
            ).to(torch.uint8)

            processor_obs = {
                "image": {
                    "base_0_rgb": base_image,
                    "left_wrist_0_rgb": left_wrist_image,
                    "right_wrist_0_rgb": right_wrist_image,
                },
                "state": norm_obs["observation.state"].to(torch.float32),
                "prompt": [instruction],
            }

            prep_state = prepare_state(self.vla_model.config, processor_obs)
            lang_tokens, lang_masks = prepare_language(
                self.vla_model.config, self.language_tokenizer, processor_obs
            )
            prep_images, prep_img_masks, _ = prepare_images(
                self.vla_model.config, self.image_processor, processor_obs
            )

            vlm_causal = getattr(self.vla_model.config, "vlm_causal", False)

            prep_images = prep_images.unsqueeze(0).to(
                device=device, dtype=self.torch_dtype
            )
            prep_img_masks = prep_img_masks.unsqueeze(0).to(device=device)
            lang_tokens = lang_tokens.unsqueeze(0).to(device=device)
            lang_masks = lang_masks.unsqueeze(0).to(device=device)
            prep_state = prep_state.unsqueeze(0).to(
                device=device, dtype=self.torch_dtype
            )

            prep_images_list.append(prep_images.cpu())
            prep_img_masks_list.append(prep_img_masks.cpu())
            lang_tokens_list.append(lang_tokens.cpu())
            lang_masks_list.append(lang_masks.cpu())
            prep_state_list.append(prep_state.cpu())

            if mode == "train":
                action_chunk, chains, prev_logprob, denoise_inds = (
                    self.sample_actions_rl(
                        images=prep_images,
                        img_masks=prep_img_masks,
                        lang_tokens=lang_tokens,
                        lang_masks=lang_masks,
                        state=prep_state,
                        mode="train",
                    )
                )
                rl_chains_list.append(chains.cpu())
                rl_logprobs_list.append(prev_logprob.cpu())
                rl_denoise_inds_list.append(denoise_inds.cpu())
            else:
                action_chunk = self.vla_model.model.sample_actions(
                    images=prep_images,
                    img_masks=prep_img_masks,
                    lang_tokens=lang_tokens,
                    lang_masks=lang_masks,
                    state=prep_state,
                    vlm_causal=vlm_causal,
                )

            action = (
                action_chunk.squeeze(0)[:, : self.action_dim]
                .to(torch.float32)
                .cpu()
                .numpy()
            )
            unnorm_data = self.normalizer.unnormalize({"action": action})
            action = torch.from_numpy(unnorm_data["action"]).to(torch.float32)

            actions_list.append(action.cpu())

        chunk_actions = torch.stack(actions_list, dim=0).cpu()

        if mode == "train":
            forward_inputs = {
                "chains": torch.cat(rl_chains_list, dim=0),
                "denoise_inds": torch.cat(rl_denoise_inds_list, dim=0),
                "prep_images": torch.cat(prep_images_list, dim=0),
                "prep_img_masks": torch.cat(prep_img_masks_list, dim=0),
                "lang_tokens": torch.cat(lang_tokens_list, dim=0),
                "lang_masks": torch.cat(lang_masks_list, dim=0),
                "prep_state": torch.cat(prep_state_list, dim=0),
            }

            result = {
                "prev_logprobs": torch.cat(rl_logprobs_list, dim=0).to(torch.float32),
                "prev_values": torch.zeros((batch_size, 1), dtype=torch.float32).cpu(),
                "forward_inputs": forward_inputs,
            }
        else:
            result = {
                "prev_logprobs": None,
                "prev_values": None,
                "forward_inputs": env_obs,
            }

        return chunk_actions, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(self, **kwargs):
        """Forward pass for Actor training to compute log-probabilities.

        Args:
            **kwargs: Keyword arguments containing 'forward_inputs' from the rollout buffer.

        Returns:
            A dictionary containing 'logprobs', 'values', and 'entropy'.
        """
        forward_inputs = kwargs.get("forward_inputs", None)

        if forward_inputs is not None and "chains" in forward_inputs:
            device = next(self.parameters()).device
            dtype = self.torch_dtype

            chains = forward_inputs["chains"].to(device=device, dtype=dtype)
            denoise_inds = forward_inputs["denoise_inds"].to(device=device)
            bsize = chains.shape[0]

            prep_images = forward_inputs["prep_images"].to(device=device, dtype=dtype)
            prep_img_masks = forward_inputs["prep_img_masks"].to(device=device)
            lang_tokens = forward_inputs["lang_tokens"].to(device=device)
            lang_masks = forward_inputs["lang_masks"].to(device=device)
            prep_state = forward_inputs["prep_state"].to(device=device, dtype=dtype)

            prefix_embs, prefix_pad_masks, prefix_att_masks = (
                self.vla_model.model.embed_prefix(
                    prep_images, prep_img_masks, lang_tokens, lang_masks, False
                )
            )
            from lingbotvla.models.vla.pi0.modeling_lingbot_vla import make_att_2d_masks

            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

            _, past_key_values = self.vla_model.model.qwenvl_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
                fill_kv_cache=True,
            )

            chains_log_probs = []
            for idx in range(self.num_steps):
                denoise_ind = denoise_inds[:, idx]
                chains_pre = chains[torch.arange(bsize), denoise_ind]
                chains_next = chains[torch.arange(bsize), denoise_ind + 1]

                x_t_mean, x_t_std, _ = self.sample_mean_var_val(
                    chains_pre,
                    denoise_ind,
                    prep_state,
                    prefix_pad_masks,
                    past_key_values,
                    "train",
                    self.num_steps,
                )
                log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
                chains_log_probs.append(log_probs)

            chains_log_probs = torch.stack(chains_log_probs, dim=1)
            log_probs = chains_log_probs[
                :, :, : self.num_action_chunks, : self.action_env_dim
            ]
            log_probs = log_probs[torch.arange(bsize), denoise_inds[:, 0]]

            return {
                "logprobs": log_probs.to(torch.float32),
                "values": torch.zeros(bsize, device=device, dtype=torch.float32),
                "entropy": torch.zeros((bsize, 1), device=device, dtype=torch.float32),
            }

        else:
            return self.vla_model(**kwargs)
