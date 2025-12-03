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

import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .base_policy import BasePolicy
from .modules.nature_cnn import PlainConv, ResNetEncoder
from .modules.utils import make_mlp
from .modules.value_head import ValueHead


class CNNPolicy(BasePolicy):
    def __init__(
        self,
        # observation space and action space info
        image_keys,
        image_size,
        state_dim,
        action_dim,
        hidden_dim,
        num_action_chunks,
        add_value_head,
        backbone,
        independent_std=True,
    ):
        super().__init__()
        self.backbone = backbone  # ["plain_conv", ]
        self.image_size = image_size  # [c, h, w]
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_action_chunks = num_action_chunks
        self.in_channels = image_size[0]
        self.image_keys = image_keys

        self.state_latent_dim = 64

        self.encoders = nn.ModuleDict()
        encoder_out_dim = 0
        if self.backbone == "plane_conv":
            for key in image_keys:
                self.encoders[key] = PlainConv(
                    in_channels=self.in_channels, out_dim=256, image_size=image_size
                )  # assume image is 64x64
                encoder_out_dim += self.encoders[key].out_dim
        elif self.backbone == "resnet":
            sample_x = torch.randn(1, *image_size)
            for key in image_keys:
                self.encoders[key] = ResNetEncoder(sample_x, out_dim=256)
                encoder_out_dim += self.encoders[key].out_dim
        else:
            raise NotImplementedError
        self.state_proj = make_mlp(
            in_channels=self.state_dim,
            mlp_channels=[
                self.state_latent_dim,
            ],
            act_builder=nn.Tanh,
            use_layer_norm=True,
        )

        # self.mlp = make_mlp(self.encoder.out_dim+self.state_dim, [512, 256], last_act=True)
        # self.actor_mean = nn.Linear(256, self.action_dim)
        self.mix_proj = make_mlp(
            in_channels=encoder_out_dim + self.state_latent_dim,
            mlp_channels=[256, 256],
            act_builder=nn.Tanh,
            use_layer_norm=True,
        )

        self.actor_mean = nn.Linear(256, self.action_dim)

        if add_value_head:
            self.value_head = ValueHead(
                input_dim=256, hidden_sizes=(256, 256, 256), activation="relu"
            )

        self.independent_std = independent_std

        if independent_std:
            self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)
        else:
            self.actor_logstd = nn.Linear(256, action_dim)

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed_env_obs = {}
        for key, value in env_obs.items():
            if key == "images":
                continue
            if value is not None:
                if isinstance(value, torch.Tensor):
                    processed_env_obs[key] = value.clone().to(device)
                else:
                    processed_env_obs[key] = value

        images = env_obs.get("images")
        if images is None:
            available_keys = list(env_obs.keys())
            raise ValueError(
                f"env_obs['images'] is None. Images are required for CNN policy. "
                f"Available keys in env_obs: {available_keys}"
            )

        if isinstance(images, dict):
            for key, value in images.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        value = value.to(device)

                        if value.dim() == 4:
                            if value.shape[-1] in [1, 3, 4] and value.shape[1] not in [
                                1,
                                3,
                                4,
                            ]:
                                value = value.permute(0, 3, 1, 2)

                        if value.dtype == torch.uint8:
                            value = value.float() / 255.0
                        elif value.dtype != torch.float32:
                            value = value.float()

                        if value.dim() != 4:
                            raise ValueError(
                                f"Expected 4D tensor [B, C, H, W], got {value.dim()}D tensor with shape {value.shape}"
                            )

                        processed_env_obs[f"images/{key}"] = value
        elif isinstance(images, torch.Tensor):
            if len(self.image_keys) == 0:
                raise ValueError(
                    "image_keys is empty. Cannot convert tensor to dict format."
                )
            key = self.image_keys[0]

            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            elif images.dtype != torch.float32:
                images = images.float()

            if images.dim() == 4 and images.shape[-1] == 3:
                processed_env_obs[f"images/{key}"] = (
                    images.clone().to(device).permute(0, 3, 1, 2)
                )
            else:
                processed_env_obs[f"images/{key}"] = images.clone().to(device)
        else:
            raise ValueError(
                f"Unsupported images type: {type(images)}. Expected dict or torch.Tensor, got {type(images)}."
            )

        return processed_env_obs

    def get_feature(self, obs, detach_encoder=False):
        visual_features = []
        for key in self.image_keys:
            visual_features.append(self.encoders[key](obs[f"images/{key}"]))
        visual_feature = torch.cat(visual_features, dim=-1)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        state_embed = self.state_proj(obs["states"])
        x = torch.cat([visual_feature, state_embed], dim=1)
        return x, visual_feature

    def default_forward(
        self,
        data,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        sample_action=False,
        **kwargs,
    ):
        obs = {}
        images = {}
        for key, value in data.items():
            if key.startswith("obs/"):
                stripped_key = key[len("obs/") :]
                if stripped_key.startswith("images/"):
                    # Extract camera name from "images/camera_name"
                    camera_name = stripped_key[len("images/") :]
                    images[camera_name] = value
                else:
                    obs[stripped_key] = value

        # Reconstruct images dict structure expected by preprocess_env_obs
        if images:
            obs["images"] = images

        action = data["action"]

        processed_obs = self.preprocess_env_obs(obs)
        full_feature, visual_feature = self.get_feature(processed_obs)
        mix_feature = self.mix_proj(full_feature)
        action_mean = self.actor_mean(mix_feature)
        if self.independent_std:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd(mix_feature)

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

    def predict_action_batch(
        self,
        env_obs,
        calulate_logprobs=True,
        calulate_values=True,
        return_obs=True,
        return_action_type="numpy_chunk",
        return_shared_feature=False,
        **kwargs,
    ):
        ############
        processed_obs = self.preprocess_env_obs(env_obs)
        full_feature, visual_feature = self.get_feature(processed_obs)
        mix_feature = self.mix_proj(full_feature)
        action_mean = self.actor_mean(mix_feature)
        if self.independent_std:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd(mix_feature)

        action_std = action_logstd.exp()
        probs = torch.distributions.Normal(action_mean, action_std)
        action = probs.rsample()  # for reparameterization trick (mean + std * N(0,1))
        chunk_logprobs = probs.log_prob(action)

        if return_action_type == "numpy_chunk":
            chunk_actions = action.reshape(-1, self.num_action_chunks, self.action_dim)
            chunk_actions = chunk_actions.cpu().numpy()
        elif return_action_type == "torch_flatten":
            chunk_actions = action.clone()
        else:
            raise NotImplementedError

        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(mix_feature)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])
        forward_inputs = {"action": action}
        if return_obs:
            for key, value in env_obs.items():
                forward_inputs[f"obs/{key}"] = value

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = full_feature
        return chunk_actions, result
