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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.modules.utils import get_act_func


class IQLMLPPolicy(MLPPolicy):
    """IQL-specific policy derived from the generic MLPPolicy."""

    def __init__(
        self,
        obs_dim,
        action_dim,
        num_action_chunks,
        add_value_head,
        add_q_head,
        q_head_type="default",
    ):
        super().__init__(
            obs_dim=obs_dim,
            action_dim=action_dim,
            num_action_chunks=num_action_chunks,
            add_value_head=add_value_head,
            add_q_head=add_q_head,
            q_head_type=q_head_type,
        )
        self.iql_type = None

    def configure_iql(self, iql_config: dict) -> None:
        hidden_dims = iql_config.get("hidden_dims", None)
        if hidden_dims is None:
            raise ValueError("hidden_dims must be provided in iql_config.")
        self.iql_type = iql_config.get("type", iql_config.get("kind", None))

        self._init_iql(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            dropout_rate=iql_config.get("dropout_rate", None),
            log_std_min=float(iql_config.get("log_std_min", -5.0)),
            log_std_max=float(iql_config.get("log_std_max", 2.0)),
            state_dependent_std=bool(iql_config.get("state_dependent_std", False)),
        )

    def _build_mlp(
        self,
        input_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        output_dim: int,
        dropout_rate: float | None = None,
        activate_final: bool = False,
    ) -> nn.Sequential:
        act = get_act_func("relu")
        layers: list[nn.Module] = []
        dims = [input_dim, *hidden_dims]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
            if dropout_rate is not None:
                layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-1], output_dim))
        if activate_final:
            layers.append(act())
        return nn.Sequential(*layers)

    def _init_iql(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | tuple[int, ...],
        dropout_rate: float | None,
        log_std_min: float,
        log_std_max: float,
        state_dependent_std: bool,
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        if self.iql_type == "actor":
            self.backbone = self._build_mlp(
                input_dim=obs_dim,
                hidden_dims=hidden_dims,
                output_dim=hidden_dims[-1],
                dropout_rate=dropout_rate,
                activate_final=True,
            )
            self.actor_mean = nn.Linear(hidden_dims[-1], action_dim)
            self.state_dependent_std = state_dependent_std
            if state_dependent_std:
                self.actor_logstd = nn.Linear(hidden_dims[-1], action_dim)
            else:
                # Keep shape aligned with generic MLPPolicy for seamless
                # actor->rollout state_dict loading in offline eval.
                self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))
            self.logstd_range = (log_std_min, log_std_max)
        elif self.iql_type == "critic":
            self.net = self._build_mlp(
                input_dim=obs_dim + action_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
            )
        elif self.iql_type == "value":
            self.net = self._build_mlp(
                input_dim=obs_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
            )
        else:
            raise ValueError(f"Unsupported iql_type: {self.iql_type}")

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.IQL:
            return self.iql_forward(**kwargs)
        return super().forward(forward_type=forward_type, **kwargs)

    def iql_forward(self, **kwargs) -> torch.Tensor:
        observations = kwargs.get("observations")
        if observations is None:
            raise ValueError("IQL forward expects observations.")
        if self.iql_type == "actor":
            return self.iql_forward_actor(**kwargs)
        if self.iql_type == "critic":
            return self.iql_forward_critic(**kwargs)
        if self.iql_type == "value":
            return self.iql_forward_value(**kwargs)
        raise RuntimeError(f"Unsupported iql_type: {self.iql_type}")

    def iql_forward_actor(self, **kwargs) -> torch.Tensor:
        observations = kwargs.get("observations")
        if observations is None:
            raise ValueError("IQL actor forward expects observations.")
        actions = kwargs.get("actions")
        temperature = float(kwargs.get("temperature", 1.0))

        feat = self.backbone(observations)
        action_mean_raw = self.actor_mean(feat)
        if self.state_dependent_std:
            action_logstd = self.actor_logstd(feat)
        else:
            action_logstd = self.actor_logstd
        action_logstd = torch.clamp(
            action_logstd, self.logstd_range[0], self.logstd_range[1]
        )
        action_std = torch.exp(action_logstd)

        if actions is not None:
            raw_actions = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
            log_prob = Normal(action_mean_raw, action_std).log_prob(raw_actions).sum(-1)
            log_prob -= (
                2 * (math.log(2) - raw_actions - F.softplus(-2 * raw_actions))
            ).sum(-1)
            return log_prob

        mode = "eval" if float(temperature) == 0.0 else "train"
        if mode == "train":
            sampling_std = action_std * max(float(temperature), 1e-6)
            raw_action = Normal(action_mean_raw, sampling_std).rsample()
        else:
            raw_action = action_mean_raw.clone()
        return torch.tanh(raw_action)

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
        """Generate actions with the IQL actor path for rollout/eval consistency."""
        env_obs = self.preprocess_env_obs(env_obs=env_obs)
        states = env_obs["states"]
        temperature = kwargs.get("temperature", None)
        if temperature is None:
            temperature = 0.0 if mode == "eval" else 1.0
        temperature = float(temperature)

        action = self.iql_forward_actor(observations=states, temperature=temperature)
        chunk_actions = (
            action.reshape(-1, self.num_action_chunks, self.action_dim).cpu().numpy()
        )

        if calculate_logprobs:
            log_probs = self.iql_forward_actor(observations=states, actions=action)
            chunk_logprobs = log_probs.unsqueeze(-1).expand(-1, self.action_dim)
        else:
            chunk_logprobs = torch.zeros(
                (states.shape[0], self.action_dim),
                device=states.device,
                dtype=states.dtype,
            )

        if hasattr(self, "value_head") and calculate_values:
            chunk_values = self.value_head(states)
        else:
            chunk_values = torch.zeros(
                (states.shape[0], 1), device=states.device, dtype=states.dtype
            )

        forward_inputs = {"action": action}
        if return_obs:
            forward_inputs["states"] = states

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def iql_forward_critic(self, **kwargs) -> torch.Tensor:
        observations = kwargs.get("observations")
        actions = kwargs.get("actions")
        if observations is None or actions is None:
            raise ValueError("IQL critic forward expects observations and actions.")
        x = torch.cat([observations, actions], dim=-1)
        return self.net(x).squeeze(-1)

    def iql_forward_value(self, **kwargs) -> torch.Tensor:
        observations = kwargs.get("observations")
        if observations is None:
            raise ValueError("IQL value forward expects observations.")
        return self.net(observations).squeeze(-1)
