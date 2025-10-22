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
import numpy as np

from .modules.utils import layer_init
from .modules.value_head import ValueHead
from .modules.q_value_head import DoubleQValueHead
from .base_policy import BasePolicy

LOG_STD_MAX = 2
LOG_STD_MIN = -5
# For SAC state-based PickCube
class MLPPolicy2(BasePolicy):
    def __init__(
            self, 
            obs_dim, action_dim, 
            hidden_dim, num_action_chunks,
            add_value_head, add_q_value_head=False, 
            ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # self.hidden_dim = hidden_dim
        self.num_action_chunks = num_action_chunks

        if add_value_head:
            raise NotImplementedError
        if add_q_value_head:
            self.q_value_head = DoubleQValueHead(
                hidden_size=obs_dim,
                action_dim=action_dim,
                use_separate_processing=False
            )

        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            
        )
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_logstd = nn.Linear(256, action_dim)

        # this should be loaded from cfg
        h, l = 1.0, -1.0
        self.register_buffer("action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict
    
    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)
    
    def _preprocess_obs(self, env_obs):
        return env_obs["states"].to("cuda")
    
    def forward(
            self,
            env_obs,
            **kwargs
        ):

        obs = self._preprocess_obs(env_obs)
        feat = self.backbone(obs)
        action_mean = self.fc_mean(feat)
        action_logstd = self.fc_logstd(feat)
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

        return action, chunk_logprobs

    def predict_action_batch(
            self, env_obs,
            calulate_logprobs=True,
            calulate_values=True,
            return_action_type="numpy_chunk", 
            **kwargs
        ):
        obs = self._preprocess_obs(env_obs)
        feat = self.backbone(obs)
        action_mean = self.fc_mean(feat)
        action_logstd = self.fc_logstd(feat)
        action_logstd = torch.tanh(action_logstd)
        action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)

        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        raw_action = probs.rsample()
        

        # for sac
        action_normalized = torch.tanh(raw_action)
        # action_normalized = raw_action
        action = action_normalized * self.action_scale + self.action_bias
        
        chunk_logprobs = probs.log_prob(raw_action)
        # print(f"raw: {chunk_logprobs.mean()}, {chunk_logprobs.sum(dim=-1).mean()}")
        chunk_logprobs = chunk_logprobs - torch.log(self.action_scale * (1 - action_normalized.pow(2)) + 1e-6)
        # print(f"scale: {chunk_logprobs.mean()}, {chunk_logprobs.sum(dim=-1).mean()}")

        if return_action_type == "numpy_chunk":
            chunk_actions = action.reshape(-1, self.num_action_chunks, self.action_dim)
            chunk_actions = chunk_actions.cpu().numpy()
        elif return_action_type == "torch_flatten":
            chunk_actions = action.clone()
        else:
            raise NotImplementedError
        
        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(obs)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        forward_inputs = {
            "obs": obs,
            "action": action
        }
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
    
    def get_q_values(self, raw_obs, actions):
        obs = self._preprocess_obs(raw_obs)
        return self.q_value_head(obs, actions)

# For PPO state-based PickCube
class MLPPolicy(BasePolicy):
    def __init__(
            self, 
            obs_dim, action_dim, 
            hidden_dim, num_action_chunks, add_value_head, add_q_value_head=False):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # self.hidden_dim = hidden_dim
        self.num_action_chunks = num_action_chunks

        if add_value_head:
            self.value_head = ValueHead(obs_dim, hidden_sizes=(256, 256, 256), activation="tanh")
        if add_q_value_head:
            self.q_value_head = DoubleQValueHead(
                hidden_size=obs_dim,
                action_dim=action_dim,
            )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def _preprocess_obs(self, env_obs):
        return env_obs["states"].to("cuda")
    
    def forward(
            self,
            data,
            compute_logprobs=True,
            compute_entropy=True,
            compute_values=True,
            sample_actions=False, 
            **kwargs
        ):
        obs = data["obs"]
        action = data["action"]

        action_mean = self.actor_mean(obs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
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
                values = self.value_head(obs)
                output_dict.update(values=values)
            elif getattr(self, "q_value_head", None):
                if not sample_actions:
                    q1_values, q2_values = self.q_value_head(obs, action)
                else:
                    new_action = probs.sample()
                    q1_values, q2_values = self.q_value_head(obs, new_action)
                output_dict.update(q1_values=q1_values)
                output_dict.update(q2_values=q2_values)
            else:
                raise NotImplementedError
        return output_dict

    def predict_action_batch(
            self, env_obs,
            calulate_logprobs=True,
            calulate_values=True,
            return_action_type="numpy_chunk", 
            **kwargs
        ):
        obs = self._preprocess_obs(env_obs)
        action_mean = self.actor_mean(obs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.rsample()

        if return_action_type == "numpy_chunk":
            chunk_actions = action.reshape(-1, self.num_action_chunks, self.action_dim)
            chunk_actions = chunk_actions.cpu().numpy()
        elif return_action_type == "torch_flatten":
            chunk_actions = action.clone()
        else:
            raise NotImplementedError
        
        chunk_logprobs = probs.log_prob(action)

        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(obs)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        forward_inputs = {
            "obs": obs,
            "action": action
        }
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
    
    def get_q_values(self, raw_obs, actions):
        obs = self._preprocess_obs(raw_obs)
        return self.q_value_head(obs, actions)


class SharedBackboneMLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, num_action_chunks, add_value_head):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_action_chunks = num_action_chunks

        self.obs_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, hidden_dim)),
            nn.ReLU()
        )
        self.action_head = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.action_dim))
        )

        self.actor_logstd = nn.Parameter(torch.ones(1, self.action_dim) * -0.5)

        if add_value_head:
            self.value_head = ValueHead(
                hidden_dim,
                hidden_sizes=(256, ),
                activation="relu"
            )

    def _preprocess_obs(self, env_obs):
        return env_obs["states"].to("cuda")

    def predict_action(self, env_obs, mode, **kwargs):
        obs = self._preprocess_obs(env_obs)
        feat = self.obs_encoder(obs)
        action_mean = self.action_head(feat)
        if mode == "eval":
            return action_mean

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()

    def forward(
            self,
            data,
            compute_logprobs=True,
            compute_entropy=True,
            compute_values=True,
            **kwargs
        ):
        obs = data["obs"]
        action = data["action"]

        feat = self.obs_encoder(obs)
        action_mean = self.action_head(feat)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if compute_logprobs:
            logprobs = probs.log_prob(action)
        if compute_entropy:
            entropy = probs.entropy()
        if compute_values:
            values = self.value_head(feat)
        return {
            "logprobs": logprobs,
            "values": values,
            "entropy": entropy
        }

    def predict_action_batch(
            self, env_obs,
            calulate_logprobs=True,
            calulate_values=True,
            **kwargs
        ):
        obs = self._preprocess_obs(env_obs)
        feat = self.obs_encoder(obs)
        action_mean = self.action_head(feat)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()

        chunk_actions = action.reshape(-1, self.num_action_chunks, self.action_dim).cpu().numpy()
        chunk_logprobs = probs.log_prob(action)

        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(feat)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        forward_inputs = {
            "obs": obs,
            "action": action
        }
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
