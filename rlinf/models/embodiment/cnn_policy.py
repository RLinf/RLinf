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

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from .modules.nature_cnn import NatureCNN, PlainConv, ResNetEncoder
from .modules.utils import layer_init, make_mlp, LOG_STD_MAX, LOG_STD_MIN
from .modules.value_head import ValueHead
from .modules.q_head import MultiQHead
from .base_policy import BasePolicy

class CNNPolicy(BasePolicy):
    def __init__(
            self,
            # observation space and action space info
            image_keys, image_size, state_dim, action_dim, 
            hidden_dim, num_action_chunks, 
            add_value_head, add_q_head,
            backbone, 
            independent_std=True, final_tanh=False, action_scale=None
        ):
        super().__init__()
        self.backbone = backbone # ["plain_conv", ]
        self.image_size = image_size # [c, h, w]
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
                ) # assume image is 64x64
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
            mlp_channels=[self.state_latent_dim, ], 
            act_builder=nn.Tanh, 
            use_layer_norm=True
        )
        
        # self.mlp = make_mlp(self.encoder.out_dim+self.state_dim, [512, 256], last_act=True)
        # self.actor_mean = nn.Linear(256, self.action_dim)
        self.mix_proj = make_mlp(
            in_channels=encoder_out_dim+self.state_latent_dim, 
            mlp_channels=[256, 256],  
            act_builder=nn.Tanh, 
            use_layer_norm=True
        )

        self.actor_mean = nn.Linear(256, self.action_dim) 

        assert add_value_head + add_q_head <= 1
        if add_value_head:
            self.value_head = ValueHead(
                input_dim=256, 
                hidden_sizes=(256, 256, 256), 
                activation="relu"
            )
        if add_q_head:
            independent_std = False
            action_scale = 1, -1
            final_tanh = True
            self.q_head = MultiQHead(
                # hidden_size=self.encoder.out_dim+self.state_dim,
                hidden_size=encoder_out_dim+self.state_latent_dim,
                hidden_dims=[256, 256, 256], 
                num_q_heads=2, 
                action_dim=action_dim,
                use_separate_processing=False
            )
        
        self.independent_std = independent_std
        self.final_tanh = final_tanh

        if independent_std:
            self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)
        else:
            self.actor_logstd = nn.Linear(256, action_dim)

        if action_scale is not None:
            h, l = action_scale
            self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
            self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        else:
            self.action_scale = None


    # def preprocess_env_obs(self, env_obs):
    #     device = next(self.parameters()).device
    #     processed_env_obs = {}
    #     for key, value in env_obs.items():
    #         if key == "images":
    #             continue
    #         if value is not None:
    #             if isinstance(value, torch.Tensor):
    #                 processed_env_obs[key] = value.clone().to(device)
    #             else:
    #                 processed_env_obs[key] = value
    #     for key, value in env_obs["images"].items():
    #         processed_env_obs[f"images/{key}"] = value.clone().to(device).permute(0, 3, 1, 2)
    #     return processed_env_obs
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
        
        # 处理 images，支持多种格式
        images = env_obs.get("images")
        if images is None:
            # 提供更详细的错误信息，帮助调试
            available_keys = list(env_obs.keys())
            raise ValueError(
                f"env_obs['images'] is None. Images are required for CNN policy. "
                f"Available keys in env_obs: {available_keys}"
            )
        
        if isinstance(images, dict):
            # 如果 images 已经是字典格式
            for key, value in images.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        # 确保在正确的设备上
                        value = value.to(device)
                        
                        # 检查是否需要 permute: 如果是 [B, H, W, C] 格式，需要转换为 [B, C, H, W]
                        if value.dim() == 4:
                            # 检查最后一个维度是否是通道维度（通常是 3 或 4）
                            if value.shape[-1] in [1, 3, 4] and value.shape[1] not in [1, 3, 4]:
                                # [B, H, W, C] 格式，需要 permute
                                value = value.permute(0, 3, 1, 2)
                        
                        # 转换为 float32 并归一化到 [0, 1] 范围（如果输入是 uint8）
                        if value.dtype == torch.uint8:
                            value = value.float() / 255.0
                        elif value.dtype != torch.float32:
                            value = value.float()
                        
                        # 确保是 [B, C, H, W] 格式
                        if value.dim() != 4:
                            raise ValueError(f"Expected 4D tensor [B, C, H, W], got {value.dim()}D tensor with shape {value.shape}")
                        
                        processed_env_obs[f"images/{key}"] = value
        elif isinstance(images, torch.Tensor):
            # 如果 images 是张量（单个相机），转换为字典格式
            if len(self.image_keys) == 0:
                raise ValueError("image_keys is empty. Cannot convert tensor to dict format.")
            # 使用第一个 image_key
            key = self.image_keys[0]
            
            # 转换为 float32 并归一化到 [0, 1] 范围（如果输入是 uint8）
            if images.dtype == torch.uint8:
                images = images.float() / 255.0
            elif images.dtype != torch.float32:
                images = images.float()
            
            # 检查是否需要 permute: maniskill_env 返回的已经是 [B, C, H, W] 格式
            # 但如果遇到 [B, H, W, C] 格式（最后一个维度是通道），需要 permute
            if images.dim() == 4 and images.shape[-1] == 3:
                # [B, H, W, C] 格式，需要 permute
                processed_env_obs[f"images/{key}"] = images.clone().to(device).permute(0, 3, 1, 2)
            else:
                # 已经是 [B, C, H, W] 格式（maniskill_env 返回的格式）
                processed_env_obs[f"images/{key}"] = images.clone().to(device)
        else:
            raise ValueError(f"Unsupported images type: {type(images)}. Expected dict or torch.Tensor, got {type(images)}.")
        
        return processed_env_obs
    
    def get_feature_0(self, obs, detach_encoder=False):
        visual_features = []
        for key in self.image_keys:
            visual_features.append(self.encoders[key](obs[f"images/{key}"]))
        visual_feature = torch.cat(visual_features, dim=-1)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        # 兼容 "states" 和 "state" 两种键名
        state_key = "states" if "states" in obs else "state"
        x = torch.cat([visual_feature, obs[state_key]], dim=1)
        return self.mlp(x), visual_feature
    
    def get_feature(self, obs, detach_encoder=False):
        visual_features = []
        for key in self.image_keys:
            visual_features.append(self.encoders[key](obs[f"images/{key}"]))
        visual_feature = torch.cat(visual_features, dim=-1)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        # 兼容 "states" 和 "state" 两种键名
        if "states" in obs:
            state_key = "states"
        elif "state" in obs:
            state_key = "state"
        else:
            raise KeyError(f"Neither 'states' nor 'state' found in obs. Available keys: {list(obs.keys())}")
        state_embed = self.state_proj(obs[state_key])
        x = torch.cat([visual_feature, state_embed], dim=1)
        return x, visual_feature

    def default_forward(
            self, 
            data, 
            compute_logprobs=True, 
            compute_entropy=True, 
            compute_values=True, 
            sample_action=False, 
            **kwargs
        ):
        obs = dict()
        for key, value in data.items():
            if key.startswith("obs/"):
                obs[key[len("obs/"):]] = value

        action = data["action"]

        full_feature, visual_feature = self.get_feature(obs)
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
    
    def sac_forward_0(
        self, obs, **kwargs
    ):
        
        x, visual_feature = self.get_feature(obs)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd(x)
        action_logstd = torch.tanh(action_logstd)
        action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)

        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        raw_action = probs.rsample()
        

        # for sac
        action_normalized = torch.tanh(raw_action)
        action = action_normalized * self.action_scale + self.action_bias
        
        chunk_logprobs = probs.log_prob(raw_action)
        chunk_logprobs = chunk_logprobs - torch.log(self.action_scale * (1 - action_normalized.pow(2)) + 1e-6)

        return action, chunk_logprobs, visual_feature
    
    def sac_forward(
        self, obs, **kwargs
    ):
        
        full_feature, visual_feature = self.get_feature(obs)
        mix_feature = self.mix_proj(full_feature)
        action_mean = self.actor_mean(mix_feature)
        action_logstd = self.actor_logstd(mix_feature)
        action_logstd = torch.tanh(action_logstd)
        action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)

        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        raw_action = probs.rsample()
        

        # for sac
        action_normalized = torch.tanh(raw_action)
        action = action_normalized * self.action_scale + self.action_bias
        
        chunk_logprobs = probs.log_prob(raw_action)
        chunk_logprobs = chunk_logprobs - torch.log(self.action_scale * (1 - action_normalized.pow(2)) + 1e-6)

        return action, chunk_logprobs, full_feature

    def predict_action_batch_0(
            self, env_obs, 
            calulate_logprobs=True,
            calulate_values=True,
            return_obs=True, 
            return_action_type="numpy_chunk", 
            return_shared_feature=False, 
            **kwargs
        ):
        x, visual_feature = self.get_feature(env_obs)
        action_mean = self.actor_mean(x)
        if self.independent_std:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd(x)

        if self.final_tanh:
            action_logstd = torch.tanh(action_logstd)
            action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)  # From SpinUp / Denis Yarats

        action_std = action_logstd.exp()
        probs = torch.distributions.Normal(action_mean, action_std)
        raw_action = probs.rsample()  # for reparameterization trick (mean + std * N(0,1))
        chunk_logprobs = probs.log_prob(raw_action)
        if self.action_scale is not None:
            action_normalized = torch.tanh(raw_action)
            action = action_normalized * self.action_scale + self.action_bias

            chunk_logprobs = chunk_logprobs - torch.log(self.action_scale * (1 - action_normalized.pow(2)) + 1e-6)
        else:
            action = raw_action

        if return_action_type == "numpy_chunk":
            chunk_actions = action.reshape(-1, self.num_action_chunks, self.action_dim)
            chunk_actions = chunk_actions.cpu().numpy()
        elif return_action_type == "torch_flatten":
            chunk_actions = action.clone()
        else:
            raise NotImplementedError
        
        if hasattr(self, "value_head") and calulate_values:
            raise NotImplementedError
            chunk_values = self.value_head(env_obs)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])
        forward_inputs = {
            "action": action
        }
        if return_obs:
            for key, value in env_obs.items():
                forward_inputs[f"obs/{key}"] = value
        
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = visual_feature
        return chunk_actions, result
    
    def predict_action_batch(
            self, env_obs, 
            calulate_logprobs=True,
            calulate_values=True,
            return_obs=True, 
            return_action_type="numpy_chunk", 
            return_shared_feature=False, 
            **kwargs
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

        if self.final_tanh:
            action_logstd = torch.tanh(action_logstd)
            action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)  # From SpinUp / Denis Yarats

        action_std = action_logstd.exp()
        probs = torch.distributions.Normal(action_mean, action_std)
        raw_action = probs.rsample()  # for reparameterization trick (mean + std * N(0,1))
        chunk_logprobs = probs.log_prob(raw_action)
        if self.action_scale is not None:
            action_normalized = torch.tanh(raw_action)
            action = action_normalized * self.action_scale + self.action_bias

            chunk_logprobs = chunk_logprobs - torch.log(self.action_scale * (1 - action_normalized.pow(2)) + 1e-6)
        else:
            action = raw_action

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
        forward_inputs = {
            "action": action
        }
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
    
    def get_q_values_0(self, obs, actions, shared_feature=None, detach_encoder=False):
        if shared_feature is None:
            shared_feature = self.encoder(obs["images"])
        if detach_encoder:
            shared_feature = shared_feature.detach()
        # 兼容 "states" 和 "state" 两种键名
        state_key = "states" if "states" in obs else "state"
        x = torch.cat([shared_feature, obs[state_key]], dim=1)
        return self.q_head(x, actions)

    
    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.q_head(shared_feature, actions)
    

class Agent(nn.Module):
    def __init__(self, obs_dims, action_dim, num_action_chunks, add_value_head):
        super().__init__()
        self.num_action_chunks = num_action_chunks
        self.feature_net = NatureCNN(obs_dims=obs_dims)  # obs_dims: dict{img_dims=(c, h, w), state_dim=}

        latent_size = self.feature_net.out_features
        if add_value_head:
            self.value_head = ValueHead(
                layer_init(nn.Linear(latent_size, 512)),
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(512, 1)),
            )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        x = self.feature_net(x)
        return self.value_head(x)

    def get_action(self, x, deterministic=False):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.value_head(x)
