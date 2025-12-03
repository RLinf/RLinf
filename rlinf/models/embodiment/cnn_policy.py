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
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import warnings

from .modules.nature_cnn import NatureCNN, PlainConv, ResNetEncoder
from .modules.utils import layer_init, make_mlp, init_mlp_weights
from .modules.value_head import ValueHead
from .modules.q_head import MultiQHead, MultiCrossQHead
from .base_policy import BasePolicy

@dataclass
class CNNConfig:
    image_size: List[int] = field(default_factory=list)
    image_keys: List[str] = field(default_factory=str)
    action_dim: int = 4
    state_dim: int = 29
    num_action_chunks: int = 1
    backbone: str = "resnet"
    extra_config: Dict[str, Any] = field(default_factory=dict)
    add_value_head: bool = False
    add_q_head: bool = False
    q_head_type: str = "default"

    state_latent_dim: int = 64
    independent_std: bool = True
    action_scale = None
    final_tanh = False
    std_range = None

    def update_from_dict(self, config_dict):
        for key, value in config_dict.items():
            if hasattr(self, key):
                self.__setattr__(key, value)
            else:
                warnings.warn(f"CNNConfig does not contain the key {key=}")
        self._update_info()

    def _update_info(self):
        if self.add_q_head:
            self.independent_std = False
            if self.action_scale is None:
                self.action_scale = -1, 1
            self.final_tanh = True
            self.std_range = (1e-5, 5)



class CNNPolicy(BasePolicy):
    def __init__(
            self, cfg: CNNConfig
        ):
        super().__init__()
        self.cfg = cfg
        self.in_channels = self.cfg.image_size[0]

        self.encoders = nn.ModuleDict()
        encoder_out_dim = 0
        if self.cfg.backbone == "plane_conv":
            for key in self.cfg.image_keys:
                self.encoders[key] = PlainConv(
                    in_channels=self.in_channels, out_dim=256, image_size=self.cfg.image_size
                ) # assume image is 64x64
                encoder_out_dim += self.encoders[key].out_dim
        elif self.cfg.backbone == "resnet":
            sample_x = torch.randn(1, *self.cfg.image_size)
            for key in self.cfg.image_keys:
                self.encoders[key] = ResNetEncoder(
                    sample_x, out_dim=256, model_cfg=self.cfg.extra_config
                )
                encoder_out_dim += self.encoders[key].out_dim
        else:
            raise NotImplementedError
        self.state_proj = nn.Sequential(
            *make_mlp(
                in_channels=self.cfg.state_dim,
                mlp_channels=[self.cfg.state_latent_dim, ], 
                act_builder=nn.Tanh, 
                use_layer_norm=True
            )
        )
        init_mlp_weights(self.state_proj, nonlinearity="tanh")
        
        # self.mlp = make_mlp(self.encoder.out_dim+self.state_dim, [512, 256], last_act=True)
        # self.actor_mean = nn.Linear(256, self.action_dim)
        self.mix_proj = nn.Sequential(
            *make_mlp(
                in_channels=encoder_out_dim+self.cfg.state_latent_dim, 
                mlp_channels=[256, 256],  
                act_builder=nn.Tanh, 
                use_layer_norm=True
            )
        )
        init_mlp_weights(self.mix_proj, nonlinearity="tanh")

        self.actor_mean = layer_init(nn.Linear(256, self.cfg.action_dim), std=0.01*np.sqrt(2))

        assert self.cfg.add_value_head + self.cfg.add_q_head <= 1
        if self.cfg.add_value_head:
            self.value_head = ValueHead(
                input_dim=256, 
                hidden_sizes=(256, 256, 256), 
                activation="relu"
            )
        if self.cfg.add_q_head:
            if self.cfg.q_head_type == "default":
                self.q_head = MultiQHead(
                    # hidden_size=self.encoder.out_dim+self.state_dim,
                    hidden_size=encoder_out_dim+self.cfg.state_latent_dim,
                    hidden_dims=[256, 256, 256], 
                    num_q_heads=2, 
                    action_feature_dim=self.cfg.action_dim
                )
            elif self.cfg.q_head_type == "crossq":
                self.q_head = MultiCrossQHead(
                    hidden_size=encoder_out_dim+self.cfg.state_latent_dim,
                    hidden_dims=[256, 256, 256], 
                    num_q_heads=2, 
                    action_feature_dim=self.cfg.action_dim
                )
        if self.cfg.independent_std:
            self.actor_logstd = nn.Parameter(torch.ones(1, self.cfg.action_dim) * -0.5)
        else:
            self.actor_logstd = layer_init(nn.Linear(256, self.cfg.action_dim))

        if self.cfg.action_scale is not None:
            l, h = self.cfg.action_scale
            self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
            self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        else:
            self.action_scale = None


    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed_env_obs = {}
        processed_env_obs["states"] = env_obs["states"].clone().to(device)
        processed_env_obs["images"] = {}
        for key, value in env_obs["images"].items():
            processed_env_obs["images"][key] = value.clone().to(device).float() / 255.0
        return processed_env_obs
    
    def get_feature_0(self, obs, detach_encoder=False):
        visual_features = []
        for key in self.cfg.image_keys:
            visual_features.append(self.encoders[key](obs["images"][key]))
        visual_feature = torch.cat(visual_features, dim=-1)
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs["states"]], dim=1)
        return self.mlp(x), visual_feature
    
    def get_feature(self, obs, detach_encoder=False):
        visual_features = []
        for key in self.cfg.image_keys:
            visual_features.append(self.encoders[key](obs["images"][key]))
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
        if self.cfg.independent_std:
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
        # action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)

        action_std = torch.exp(action_logstd)
        if self.cfg.std_range is not None:
            action_std = torch.clamp(action_std, self.cfg.std_range[0], self.cfg.std_range[1])

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
        # action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)

        action_std = torch.exp(action_logstd)
        if self.cfg.std_range is not None:
            action_std = torch.clamp(action_std, self.cfg.std_range[0], self.cfg.std_range[1])

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
            # action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)  # From SpinUp / Denis Yarats

        action_std = action_logstd.exp()
        if self.std_range is not None:
            action_std = torch.clamp(action_std, self.std_range[0], self.std_range[1])

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
            mode="train", 
            **kwargs
        ):
        full_feature, visual_feature = self.get_feature(env_obs)
        mix_feature = self.mix_proj(full_feature)
        action_mean = self.actor_mean(mix_feature)
        if self.cfg.independent_std:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd(mix_feature)

        if self.cfg.final_tanh:
            action_logstd = torch.tanh(action_logstd)
            # action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)  # From SpinUp / Denis Yarats

        action_std = action_logstd.exp()
        if self.cfg.std_range is not None:
            action_std = torch.clamp(action_std, self.cfg.std_range[0], self.cfg.std_range[1])

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

            chunk_logprobs = chunk_logprobs - torch.log(self.action_scale * (1 - action_normalized.pow(2)) + 1e-6)
        else:
            action = raw_action

        if return_action_type == "numpy_chunk":
            chunk_actions = action.reshape(-1, self.cfg.num_action_chunks, self.cfg.action_dim)
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
        x = torch.cat([shared_feature, obs["states"]], dim=1)
        return self.q_head(x, actions)

    
    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
        return self.q_head(shared_feature, actions)
    
        mix_feature = self.mix_proj(shared_feature)
        if detach_encoder:
            mix_feature = mix_feature.detach()
        return self.q_head(mix_feature, actions)

    def crossq_q_forward(self, obs, actions, next_obs=None, next_actions=None, shared_feature=None, detach_encoder=False):
        if shared_feature is None:
            shared_feature, visual_feature = self.get_feature(obs)
            if next_obs is not None:
                next_shared_feature, next_visual_feature = self.get_feature(next_obs)
        if detach_encoder:
            shared_feature = shared_feature.detach()
            if next_obs is not None:
                next_shared_feature = next_shared_feature.detach()
        return self.q_head(
            shared_feature, actions, 
            next_state_features=next_shared_feature if next_obs is not None else None,
            next_action_features=next_actions
        )

    def crossq_forward(self, obs, **kwargs):
        return self.sac_forward(obs, **kwargs)
    

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
