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

import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.q_head import MultiCrossQHead, MultiQHead
from rlinf.models.embodiment.modules.resnet_utils import ResNetEncoder
from rlinf.models.embodiment.modules.utils import init_mlp_weights, layer_init, make_mlp
from rlinf.models.embodiment.modules.value_head import ValueHead


@dataclass
class CNNConfig:
    image_size: list[int] = field(default_factory=list)
    image_num: int = 1
    action_dim: int = 4
    state_dim: int = 29
    num_action_chunks: int = 1
    backbone: str = "resnet"
    model_path: Optional[str] = None
    encoder_config: dict[str, Any] = field(default_factory=dict)
    add_value_head: bool = False
    add_q_head: bool = False
    q_head_type: str = "default"

    state_latent_dim: int = 64
    independent_std: bool = True
    action_scale = None
    final_tanh = False
    std_range = None
    logstd_range = None

    num_q_heads = 2

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            self.independent_std = False
            if self.action_scale is None:
                self.action_scale = -1, 1
            self.final_tanh = True
            if self.backbone == "resnet":
                self.std_range = (1e-5, 5)

        assert self.model_path is not None, "Please specify the model_path."
        assert "ckpt_name" in self.encoder_config, (
            "Please specify the ckpt_name in encoder_config to load pretrained encoder weights."
        )
        ckpt_path = os.path.join(self.model_path, self.encoder_config["ckpt_name"])
        assert os.path.exists(ckpt_path), (
            f"Pretrained encoder weights not found at {ckpt_path} with model path {self.model_path} and encoder ckpt name {self.encoder_config['ckpt_name']}"
        )
        self.encoder_config["ckpt_path"] = ckpt_path


class CNNPolicy(nn.Module, BasePolicy):
    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        self.in_channels = self.cfg.image_size[0]
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3)
        )
        self.register_buffer(
            "img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3)
        )
        self.encoders = nn.ModuleList()
        encoder_out_dim = 0
        if self.cfg.backbone == "resnet":
            sample_x = torch.randn(1, *self.cfg.image_size)
            for img_id in range(self.cfg.image_num):
                self.encoders.append(
                    ResNetEncoder(
                        sample_x, out_dim=256, encoder_cfg=self.cfg.encoder_config
                    )
                )
                encoder_out_dim += self.encoders[img_id].out_dim
        else:
            raise NotImplementedError

        if self.cfg.backbone == "resnet":
            self.state_proj = nn.Sequential(
                *make_mlp(
                    in_channels=self.cfg.state_dim,
                    mlp_channels=[
                        self.cfg.state_latent_dim,
                    ],
                    act_builder=nn.Tanh,
                    last_act=True,
                    use_layer_norm=True,
                )
            )
            self.state_proj._fsdp_wrap_name = "state_proj"
            init_mlp_weights(self.state_proj, nonlinearity="tanh")
            self.mix_proj = nn.Sequential(
                *make_mlp(
                    in_channels=encoder_out_dim + self.cfg.state_latent_dim,
                    mlp_channels=[256, 256],
                    act_builder=nn.Tanh,
                    last_act=True,
                    use_layer_norm=True,
                )
            )
            init_mlp_weights(self.mix_proj, nonlinearity="tanh")

            self.actor_mean = layer_init(
                nn.Linear(256, self.cfg.action_dim), std=0.01 * np.sqrt(2)
            )

        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=256, hidden_sizes=(256, 256, 256), activation="relu"
            )
        if self.cfg.add_q_head:
            if self.cfg.backbone == "resnet":
                hidden_size = encoder_out_dim + self.cfg.state_latent_dim
                hidden_dims = [256, 256]
            if self.cfg.q_head_type == "default":
                self.q_head = MultiQHead(
                    hidden_size=hidden_size,
                    hidden_dims=hidden_dims,
                    num_q_heads=self.cfg.num_q_heads,
                    action_feature_dim=self.cfg.action_dim,
                )
            elif self.cfg.q_head_type == "crossq":
                self.q_head = MultiCrossQHead(
                    hidden_size=hidden_size,
                    hidden_dims=hidden_dims,
                    num_q_heads=self.cfg.num_q_heads,
                    action_feature_dim=self.cfg.action_dim,
                )
        if self.cfg.independent_std:
            self.actor_logstd = nn.Parameter(torch.ones(1, self.cfg.action_dim) * -0.5)
        else:
            self.actor_logstd = layer_init(nn.Linear(256, self.cfg.action_dim))

        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            self.register_buffer(
                "action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32)
            )
        else:
            self.action_scale = None
        self.torch_compile_enabled = False

    def _get_feature_from_processed_tensors(
        self,
        main_images: torch.Tensor,
        states: torch.Tensor,
        extra_view_images: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_features = []
        for img_id in range(self.cfg.image_num):
            if img_id == 0:
                images = main_images
            else:
                if extra_view_images is None:
                    raise ValueError(
                        "extra_view_images is required when image_num > 1."
                    )
                images = extra_view_images[:, img_id - 1]
            if images.shape[3] == 3:
                # [B, H, W, C] -> [B, C, H, W]
                images = images.permute(0, 3, 1, 2)
            visual_features.append(self.encoders[img_id](images))
        visual_feature = torch.cat(visual_features, dim=-1)
        state_embed = self.state_proj(states)
        full_feature = torch.cat([visual_feature, state_embed], dim=1)
        return full_feature, visual_feature

    def _policy_head(
        self, full_feature: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mix_feature = self.mix_proj(full_feature)
        action_mean = self.actor_mean(mix_feature)
        if self.cfg.independent_std:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd(mix_feature)
        return mix_feature, action_mean, action_logstd

    def _actor_forward_from_processed_tensors(
        self,
        main_images: torch.Tensor,
        states: torch.Tensor,
        extra_view_images: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        full_feature, _ = self._get_feature_from_processed_tensors(
            main_images=main_images,
            states=states,
            extra_view_images=extra_view_images,
        )
        mix_feature, action_mean, action_logstd = self._policy_head(full_feature)
        return full_feature, mix_feature, action_mean, action_logstd

    @property
    def num_action_chunks(self):
        return self.cfg.num_action_chunks

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        mean = self.img_mean.to(device)
        std = self.img_std.to(device)

        processed_env_obs = {}
        processed_env_obs["states"] = env_obs["states"].clone().to(device)
        x = env_obs["main_images"].clone().to(device).float() / 255.0
        processed_env_obs["main_images"] = (x - mean) / std

        if env_obs.get("extra_view_images", None) is not None:
            ex = env_obs["extra_view_images"].clone().to(device).float() / 255.0
            ex = (ex - mean.unsqueeze(1)) / std.unsqueeze(1)
            processed_env_obs["extra_view_images"] = ex

        return processed_env_obs

    def get_feature(self, obs, detach_encoder=False):
        x, visual_feature = self._get_feature_from_processed_tensors(
            main_images=obs["main_images"],
            states=obs["states"],
            extra_view_images=obs.get("extra_view_images"),
        )
        if detach_encoder:
            x = x.detach()
            visual_feature = visual_feature.detach()
        return x, visual_feature

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs", None)
        if obs is not None:
            obs = self.preprocess_env_obs(obs)
            kwargs.update({"obs": obs})
        next_obs = kwargs.get("next_obs", None)
        if next_obs is not None:
            next_obs = self.preprocess_env_obs(next_obs)
            kwargs.update({"next_obs": next_obs})

        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        elif forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        elif forward_type == ForwardType.CROSSQ:
            return self.crossq_forward(**kwargs)
        elif forward_type == ForwardType.CROSSQ_Q:
            return self.crossq_q_forward(**kwargs)
        elif forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        sample_action=False,
        **kwargs,
    ):
        obs = {
            "main_images": forward_inputs["main_images"],
            "states": forward_inputs["states"],
        }
        if "extra_view_images" in forward_inputs:
            obs["extra_view_images"] = forward_inputs["extra_view_images"]
        obs = self.preprocess_env_obs(obs)
        action = forward_inputs["action"]
        full_feature, mix_feature, action_mean, action_logstd = (
            self._actor_forward_from_processed_tensors(
                obs["main_images"],
                obs["states"],
                obs.get("extra_view_images"),
            )
        )
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        output_dict = {}
        if compute_logprobs:
            logprobs = probs.log_prob(action)
            output_dict.update(logprobs=logprobs)
        if compute_entropy:
            entropy = probs.entropy()
            output_dict.update(entropy=entropy)
        if compute_values:
            if getattr(self, "value_head", None):
                values = self.value_head(mix_feature)
                output_dict.update(values=values)
            else:
                raise NotImplementedError
        return output_dict

    def sac_forward(self, obs, **kwargs):
        full_feature, mix_feature, action_mean, action_logstd = (
            self._actor_forward_from_processed_tensors(
                obs["main_images"],
                obs["states"],
                obs.get("extra_view_images"),
            )
        )
        action_std = torch.exp(action_logstd)
        if self.cfg.std_range is not None:
            action_std = torch.clamp(
                action_std, self.cfg.std_range[0], self.cfg.std_range[1]
            )

        probs = Normal(action_mean, action_std)
        raw_action = probs.rsample()

        action_normalized = torch.tanh(raw_action)
        action = action_normalized * self.action_scale + self.action_bias

        chunk_logprobs = probs.log_prob(raw_action)
        chunk_logprobs = chunk_logprobs - torch.log(
            self.action_scale * (1 - action_normalized.pow(2)) + 1e-6
        )

        return action, chunk_logprobs, full_feature

    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        return_shared_feature=False,
        mode="train",
        **kwargs,
    ):
        obs = self.preprocess_env_obs(env_obs)
        full_feature, mix_feature, action_mean, action_logstd = (
            self._actor_forward_from_processed_tensors(
                obs["main_images"],
                obs["states"],
                obs.get("extra_view_images"),
            )
        )

        action_std = action_logstd.exp()
        if self.cfg.std_range is not None:
            action_std = torch.clamp(
                action_std, self.cfg.std_range[0], self.cfg.std_range[1]
            )

        probs = torch.distributions.Normal(action_mean, action_std)
        if mode == "train":
            raw_action = probs.sample()
        elif mode == "eval":
            raw_action = action_mean.clone()
        else:
            raise NotImplementedError(f"{mode=}")

        chunk_logprobs = probs.log_prob(raw_action)
        if self.action_scale is not None:
            action_normalized = torch.tanh(raw_action)
            action = action_normalized * self.action_scale + self.action_bias

            chunk_logprobs = chunk_logprobs - torch.log(
                self.action_scale * (1 - action_normalized.pow(2)) + 1e-6
            )
        else:
            action = raw_action

        chunk_actions = action.reshape(
            -1, self.cfg.num_action_chunks, self.cfg.action_dim
        )
        chunk_actions = chunk_actions.cpu().numpy()

        if hasattr(self, "value_head") and calculate_values:
            chunk_values = self.value_head(mix_feature)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])
        forward_inputs = {"action": action}
        if return_obs:
            forward_inputs["main_images"] = env_obs["main_images"]
            forward_inputs["states"] = env_obs["states"]
            if "extra_view_images" in env_obs:
                forward_inputs["extra_view_images"] = env_obs["extra_view_images"]

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = full_feature
        return chunk_actions, result

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.q_head(shared_feature, actions)

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
            if next_obs is not None:
                next_shared_feature, next_visual_feature = self.get_feature(next_obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
            if next_obs is not None:
                next_shared_feature = next_shared_feature.detach()
        return self.q_head(
            shared_feature,
            actions,
            next_state_features=next_shared_feature if next_obs is not None else None,
            next_action_features=next_actions,
        )

    def crossq_forward(self, obs, **kwargs):
        return self.sac_forward(obs, **kwargs)

    def enable_torch_compile(self, mode: str = "max-autotune-no-cudagraphs"):
        if self.torch_compile_enabled:
            return
        self._actor_forward_from_processed_tensors = torch.compile(
            self._actor_forward_from_processed_tensors, mode=mode
        )

        self.torch_compile_enabled = True
