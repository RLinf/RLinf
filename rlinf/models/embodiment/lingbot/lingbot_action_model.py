# Copyright 2025 The RLinf Authors.
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


class LingbotForRLActionPrediction(nn.Module, BasePolicy):
    def __init__(self, cfg, torch_dtype=torch.bfloat16):
        super().__init__()
        self.action_dim = cfg.action_dim
        self.num_action_chunks = getattr(cfg, "num_action_chunks", 50)
        self.torch_dtype = torch_dtype

        # --- 1. Load LingBot Foundation Model ---
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
                cfg,
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
            cfg,
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

    @torch.no_grad()
    def predict_action_batch(
        self, env_obs, calulate_logprobs=False, calulate_values=False, **kwargs
    ):
        batch_size = len(env_obs["task_descriptions"])
        device = next(self.parameters()).device

        actions_list = []

        def process_img(img):
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

        result = {"prev_logprobs": None, "prev_values": None, "forward_inputs": env_obs}
        return chunk_actions, result

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(self, **kwargs):
        return self.vla_model(**kwargs)
