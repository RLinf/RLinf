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

from .modules.utils import layer_init
from .modules.value_head import ValueHead


class MLPPolicy(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hidden_dim, num_action_chunks, add_value_head
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        # self.hidden_dim = hidden_dim
        self.num_action_chunks = num_action_chunks

        self.value_head = (
            ValueHead(obs_dim, hidden_sizes=(256, 256, 256), activation="tanh")
            if add_value_head
            else None
        )

        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01 * np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def preprocess_obs(self, env_obs):
        return env_obs["states"].to("cuda")

    def forward(
        self,
        data,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        obs = data["obs"]
        action = data["action"]

        action_mean = self.actor_mean(obs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        ret_dict = {}
        if compute_logprobs:
            logprobs = probs.log_prob(action)
            ret_dict["logprobs"] = logprobs
        if compute_entropy:
            entropy = probs.entropy()
            ret_dict["entropy"] = entropy
        if compute_values:
            values = self.value_head(obs)
            ret_dict["values"] = values
        return ret_dict

    def predict_action_batch(
        self, env_obs, calulate_logprobs=True, calulate_values=True, **kwargs
    ):
        obs = self.preprocess_obs(env_obs)
        action_mean = self.actor_mean(obs)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()

        chunk_actions = (
            action.reshape(-1, self.num_action_chunks, self.action_dim).cpu().numpy()
        )
        chunk_logprobs = probs.log_prob(action)

        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(obs)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        forward_inputs = {"obs": obs, "action": action}
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result


class SharedBackboneMLPPolicy(nn.Module):
    def __init__(
        self, obs_dim, action_dim, hidden_dim, num_action_chunks, add_value_head
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_action_chunks = num_action_chunks

        self.obs_encoder = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, hidden_dim)),
            nn.ReLU(),
        )
        self.action_head = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, 128)),
            nn.ReLU(),
            layer_init(nn.Linear(128, self.action_dim)),
        )

        self.actor_logstd = nn.Parameter(torch.ones(1, self.action_dim) * -0.5)

        if add_value_head:
            self.value_head = ValueHead(
                hidden_dim, hidden_sizes=(256,), activation="relu"
            )

    def preprocess_obs(self, env_obs):
        return env_obs["states"].to("cuda")

    def predict_action(self, env_obs, mode, **kwargs):
        obs = self.preprocess_obs(env_obs)
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
        **kwargs,
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
        return {"logprobs": logprobs, "values": values, "entropy": entropy}

    def predict_action_batch(
        self, env_obs, calulate_logprobs=True, calulate_values=True, **kwargs
    ):
        obs = self.preprocess_obs(env_obs)
        feat = self.obs_encoder(obs)
        action_mean = self.action_head(feat)

        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        action = probs.sample()

        chunk_actions = (
            action.reshape(-1, self.num_action_chunks, self.action_dim).cpu().numpy()
        )
        chunk_logprobs = probs.log_prob(action)

        if hasattr(self, "value_head") and calulate_values:
            chunk_values = self.value_head(feat)
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        forward_inputs = {"obs": obs, "action": action}
        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
