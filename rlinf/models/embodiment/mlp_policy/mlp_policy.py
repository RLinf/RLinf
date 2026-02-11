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

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.q_head import MultiCrossQHead, MultiQHead
from rlinf.models.embodiment.modules.utils import get_act_func, layer_init
from rlinf.models.embodiment.modules.value_head import ValueHead


class MLPPolicy(nn.Module, BasePolicy):
    def __init__(
        self,
        obs_dim,
        action_dim,
        num_action_chunks,
        add_value_head,
        add_q_head,
        q_head_type="default",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.torch_compile_enabled = False
        # default setting
        self.independent_std = True
        self.final_tanh = False
        activation = "tanh"
        action_scale = None

        assert add_value_head + add_q_head <= 1
        if add_value_head:
            self.value_head = ValueHead(
                obs_dim, hidden_sizes=(256, 256, 256), activation=activation
            )
        if add_q_head:
            self.independent_std = False
            self.final_tanh = True
            self.logstd_range = (-5, 2)
            action_scale = -1, 1
            if q_head_type == "default":
                self.q_head = MultiQHead(
                    hidden_size=obs_dim,
                    hidden_dims=[256, 256, 256],
                    num_q_heads=2,
                    action_feature_dim=action_dim,
                )
            elif q_head_type == "crossq":
                self.q_head = MultiCrossQHead(
                    hidden_size=obs_dim,
                    hidden_dims=[256, 256, 256],
                    num_q_heads=2,
                    action_feature_dim=action_dim,
                )
            else:
                raise ValueError(f"Invalid q_head_type: {q_head_type}")

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

        if self.independent_std:
            self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)
        else:
            self.actor_logstd = nn.Linear(256, action_dim)

        if action_scale is not None:
            l, h = action_scale
            self.register_buffer(
                "action_scale", torch.tensor((h - l) / 2.0, dtype=torch.float32)
            )
            self.register_buffer(
                "action_bias", torch.tensor((h + l) / 2.0, dtype=torch.float32)
            )
        else:
            self.action_scale = None

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        return {"states": env_obs["states"].to(device)}

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            obs = self.preprocess_env_obs(obs)
            kwargs.update({"obs": obs})
        next_obs = kwargs.get("next_obs")
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

    def sac_forward(self, obs, **kwargs):
        feat = self.backbone(obs["states"])
        action_mean = self.actor_mean(feat)
        action_logstd = self.actor_logstd(feat)
        action_logstd = torch.tanh(action_logstd)
        action_logstd = self.logstd_range[0] + 0.5 * (
            self.logstd_range[1] - self.logstd_range[0]
        ) * (action_logstd + 1)

        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        raw_action = probs.rsample()

        action_normalized = torch.tanh(raw_action)
        action = action_normalized * self.action_scale + self.action_bias

        chunk_logprobs = probs.log_prob(raw_action)
        chunk_logprobs = chunk_logprobs - torch.log(
            self.action_scale * (1 - action_normalized.pow(2)) + 1e-6
        )

        return action, chunk_logprobs, None

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
        **kwargs,
    ):
        states = forward_inputs["states"]
        action = forward_inputs["action"]

        feat = self.backbone(states)
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
                values = self.value_head(states)
                output_dict.update(values=values)
            else:
                raise NotImplementedError
        return output_dict

    def _sample_actions(
        self, states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(states)
        action_mean = self.actor_mean(feat)

        if self.independent_std:
            action_logstd = self.actor_logstd.expand_as(action_mean)
        else:
            action_logstd = self.actor_logstd(feat)
        if self.final_tanh:
            action_logstd = torch.tanh(action_logstd)
            action_logstd = self.logstd_range[0] + 0.5 * (
                self.logstd_range[1] - self.logstd_range[0]
            ) * (action_logstd + 1)

        return action_mean, action_logstd

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        env_obs = self.preprocess_env_obs(env_obs=env_obs)
        action_mean, action_logstd = self._sample_actions(env_obs["states"])

        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

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

        chunk_actions = action.reshape(-1, self.num_action_chunks, self.action_dim)
        chunk_actions = chunk_actions.cpu().numpy()

        if hasattr(self, "value_head") and calculate_values:
            chunk_values = self.value_head(env_obs["states"])
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        forward_inputs = {"action": action}
        if return_obs:
            forward_inputs["states"] = env_obs["states"]

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        return self.q_head(obs["states"], actions)

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
        return self.q_head(
            obs["states"],
            actions,
            next_state_features=next_obs["states"] if next_obs is not None else None,
            next_action_features=next_actions,
        )

    def crossq_forward(self, obs, **kwargs):
        return self.sac_forward(obs, **kwargs)

    def enable_torch_compile(self, mode: str = "max-autotune-no-cudagraphs"):
        if self.torch_compile_enabled:
            return

        self._sample_actions = torch.compile(self._sample_actions, mode=mode)

        self.torch_compile_enabled = True
