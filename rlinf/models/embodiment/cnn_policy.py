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

from .modules.nature_cnn import NatureCNN, PlainConv
from .modules.utils import layer_init, make_mlp, LOG_STD_MAX, LOG_STD_MIN
from .modules.value_head import ValueHead
from .modules.q_head import DoubleQHead
from .base_policy import BasePolicy

class CNNPolicy(BasePolicy):
    def __init__(
            self,
            image_size, state_dim, action_dim, 
            hidden_dim, num_action_chunks, 
            add_value_head, add_q_head,
        ):
        super().__init__()
        self.image_size = image_size # [c, h, w]
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.num_action_chunks = num_action_chunks
        self.in_channels = image_size[0]

        self.encoder = PlainConv(
            in_channels=self.in_channels, out_dim=256, image_size=image_size
        ) # assume image is 64x64
        self.mlp = make_mlp(self.encoder.out_dim+state_dim, [512, 256], last_act=True)
        self.actor_mean = nn.Linear(256, action_dim)

        if add_q_head:
            independent_std = False
            action_scale = 1, -1
            final_tanh = True
            self.q_head = DoubleQHead(
                hidden_size=self.encoder.out_dim+self.state_dim,
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


    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed_env_obs = {}
        for key, value in env_obs.items():
            if value is not None:
                if isinstance(value, torch.Tensor):
                    processed_env_obs[key] = value.clone().to(device)
                else:
                    processed_env_obs[key] = value
        processed_env_obs["images"] = processed_env_obs["images"].permute(0, 3, 1, 2)
        return processed_env_obs
    
    def get_feature(self, obs, detach_encoder=False):
        visual_feature = self.encoder(obs["images"])
        if detach_encoder:
            visual_feature = visual_feature.detach()
        x = torch.cat([visual_feature, obs["states"]], dim=1)
        return self.mlp(x), visual_feature

    def default_forward(
            self, 
            data, 
            detach_encoder=False
        ):
        raise NotImplementedError
        obs = data["obs"]
        action = data["action"]
        x, visual_feature = self.get_feature(obs, detach_encoder)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats
        return mean, log_std, visual_feature
    
    def sac_forward(
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

    def predict_action_batch(
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
            forward_inputs["obs"] = env_obs
        
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = visual_feature
        return chunk_actions, result
    
    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        if shared_feature is None:
            shared_feature = self.encoder(obs["images"])
        if detach_encoder:
            shared_feature = shared_feature.detach()
        x = torch.cat([shared_feature, obs["states"]], dim=1)
        return self.q_head(x, actions)
    

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
