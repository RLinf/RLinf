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

from .modules.utils import layer_init, get_act_func, LOG_STD_MAX, LOG_STD_MIN
from .modules.value_head import ValueHead
from .modules.q_head import MultiQHead, MultiCrossQHead
from .base_policy import BasePolicy


# For PPO state-based PickCube
class MLPPolicy(BasePolicy):
    def __init__(
            self, 
            obs_dim, action_dim, 
            hidden_dim, num_action_chunks, 
            add_value_head, add_q_head, q_head_type="default"
            ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # self.hidden_dim = hidden_dim
        self.num_action_chunks = num_action_chunks

        # default setting
        independent_std = True
        activation = "tanh" 
        action_scale = None 
        final_tanh = False

        assert add_value_head + add_q_head <=1
        if add_value_head:
            self.value_head = ValueHead(
                obs_dim, hidden_sizes=(256, 256, 256), 
                activation=activation
            )
        if add_q_head:
            independent_std = False
            action_scale = 1, -1
            final_tanh = True
            if q_head_type == "default":
                self.q_head = MultiQHead(
                    hidden_size=obs_dim,
                    hidden_dims=[256, 256, 256], 
                    num_q_heads=2, 
                    action_feature_dim=action_dim
                )
            elif q_head_type == "crossq":
                self.q_head = MultiCrossQHead(
                    hidden_size=obs_dim,
                    hidden_dims=[256, 256, 256], 
                    num_q_heads=2, 
                    action_feature_dim=action_dim
                )
            else:
                raise ValueError(f"Invalid q_head_type: {q_head_type}")

        self.final_tanh = final_tanh
        
        act = get_act_func(activation)

        self.backbone = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            act(),
            layer_init(nn.Linear(256, 256)),
            act(),
            layer_init(nn.Linear(256, 256)),
            act(),   
        )
        self.actor_mean = layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2))

        self.independent_std = independent_std
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
        return {"states": env_obs["states"].to(device)}
    
    def sac_forward(
        self, obs, **kwargs
    ):
        feat = self.backbone(obs["states"])
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd(feat)
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

        return action, chunk_logprobs, None
    
    def default_forward(
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

        feat = self.backbone(obs)
        action_mean = self.actor_mean(feat)

        if self.independent_std:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd(feat)
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
            else:
                raise NotImplementedError
        return output_dict

    def predict_action_batch(
            self, env_obs,
            calulate_logprobs=True,
            calulate_values=True,
            return_obs=True, 
            return_action_type="numpy_chunk", 
            return_shared_feature=False, 
            **kwargs
        ):
        feat = self.backbone(env_obs["states"])
        action_mean = self.actor_mean(feat)

        if self.independent_std:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd(feat)
        
        if self.final_tanh:
            action_logstd = torch.tanh(action_logstd)
            action_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (action_logstd + 1)
        
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        raw_action = probs.sample()
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
            chunk_values = self.value_head(env_obs["states"])
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])


        forward_inputs = {
            "action": action
        }
        if return_obs:
            forward_inputs["obs"] = env_obs["states"]
        
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        if return_shared_feature:
            result["shared_feature"] = None
        return chunk_actions, result
    
    def get_q_values(self, obs, actions, shared_feature=None, detach_encoder=False):
        return self.q_head(obs["states"], actions)
    
    def crossq_q_forward(self, obs, actions, next_obs=None, next_actions=None, shared_feature=None, detach_encoder=False):
        return self.q_head(
            obs["states"], actions, 
            next_state_features=next_obs["states"] if next_obs is not None else None,
            next_action_features=next_actions
        )

    def crossq_forward(self, obs, **kwargs):
        return self.sac_forward(obs, **kwargs)


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
