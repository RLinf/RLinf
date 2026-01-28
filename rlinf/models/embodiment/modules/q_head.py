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

from .batch_renorm import make_batchrenorm
from .utils import get_act_func, make_mlp


class QHead(nn.Module):
    """
    Q-value head for SAC critic networks.
    Processes state and action separately before fusion to handle dimension imbalance.

    Architecture:
        - State pathway: projects from hidden_size to 256
        - Action pathway: projects from action_dim to 256
        - Fusion: concatenate [256, 256] -> 512 -> 256 -> 128 -> 1
    """

    def __init__(
        self,
        hidden_size,
        action_feature_dim,
        hidden_dims,
        output_dim=1,
        train_action_encoder=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_feature_dim = action_feature_dim
        self.train_action_encoder = train_action_encoder

        self.nonlinearity = "tanh"

        if self.train_action_encoder:
            self.action_encoder = nn.Sequential(
                *make_mlp(
                    in_channels=action_feature_dim,
                    mlp_channels=[1024, 1024],
                )
            )
            action_hidden_dim = 1024
        else:
            action_hidden_dim = action_feature_dim

        self.net = nn.Sequential(
            *make_mlp(
                in_channels=hidden_size + action_hidden_dim,
                mlp_channels=hidden_dims
                + [
                    output_dim,
                ],
                act_builder=get_act_func(self.nonlinearity),
                use_layer_norm=True,
                last_act=False,
            )
        )

        self._init_weights(self.nonlinearity)

    def _init_weights(self, nonlinearity="relu"):
        if nonlinearity == "tanh":
            gain = nn.init.calculate_gain("tanh")
        else:
            gain = None

        for m in self.net:
            if isinstance(m, nn.Linear):
                if m is self.net[-1]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    if nonlinearity == "tanh":
                        # tanh: Xavier
                        nn.init.xavier_uniform_(m.weight, gain=gain)
                    else:
                        # relu: Kaiming
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity=nonlinearity
                        )

                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, state_features, action_features):
        """
        Forward pass for Q-value computation.

        Args:
            state_features (torch.Tensor): State representation [batch_size, hidden_size]
            action_features (torch.Tensor): Action representation [batch_size, action_dim]

        Returns:
            torch.Tensor: Q-values [batch_size, output_dim]
        """
        if self.train_action_encoder:
            action_features = self.action_encoder(action_features)

        # Original simple concatenation
        x = torch.cat([state_features, action_features], dim=-1)
        q_values = self.net(x)

        return q_values


class MultiQHead(nn.Module):
    """
    Double Q-network for SAC to reduce overestimation bias.
    """

    def __init__(
        self,
        hidden_size,
        action_feature_dim,
        hidden_dims,
        num_q_heads=2,
        output_dim=1,
        train_action_encoder=False,
    ):
        super().__init__()

        self.num_q_heads = num_q_heads
        qs = []
        for q_id in range(self.num_q_heads):
            qs.append(
                QHead(
                    hidden_size,
                    action_feature_dim,
                    hidden_dims,
                    output_dim,
                    train_action_encoder,
                )
            )
        self.qs = nn.ModuleList(qs)

    def forward(self, state_features, action_features):
        """
        Forward pass for both Q-networks.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Q1 and Q2 values
        """
        q_vs = []
        for qf in self.qs:
            q_vs.append(qf(state_features, action_features))
        return torch.cat(q_vs, dim=-1)

    def q_id_forward(self, q_id, state_features, action_features):
        """Forward pass for Q1 network only"""
        return self.qs[q_id](state_features, action_features)


class CrossQHead(nn.Module):
    """
    Q-value head with batchrenorm, for crossq critic networks.
    """

    def __init__(
        self,
        hidden_size,
        action_feature_dim,
        hidden_dims,
        output_dim=1,
        train_action_encoder=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_feature_dim = action_feature_dim
        self.train_action_encoder = train_action_encoder

        self.nonlinearity = "relu"

        if train_action_encoder:
            raise NotImplementedError
        else:
            self.net = nn.ModuleList(
                make_mlp(
                    in_channels=hidden_size + action_feature_dim,
                    mlp_channels=hidden_dims
                    + [
                        output_dim,
                    ],
                    act_builder=get_act_func(self.nonlinearity),
                    last_act=False,
                )
            )
            self.brn = nn.ModuleList(
                make_batchrenorm(
                    in_channels=hidden_size + action_feature_dim,
                    mlp_channels=hidden_dims
                    + [
                        output_dim,
                    ],
                    last_act=False,
                )
            )

        self._init_weights(self.nonlinearity)

    def _init_weights(self, nonlinearity="relu"):
        for m in self.net:
            if isinstance(m, nn.Linear):
                if m is self.net[-1]:
                    nn.init.normal_(m.weight, mean=0.0, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity=nonlinearity
                    )
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, state_features, action_features, brn_train=False):
        """
        Forward pass for Q-value with batchrenorm computation.

        Args:
            state_features (torch.Tensor): State representation [batch_size, hidden_size]
            action_features (torch.Tensor): Action representation [batch_size, action_dim]

        Returns:
            torch.Tensor: Q-values [batch_size, output_dim]
        """
        assert len(self.brn) == len(self.net)
        if self.train_action_encoder:
            raise NotImplementedError
        else:
            # Original simple concatenation
            x = torch.cat([state_features, action_features], dim=-1)
            for brn_layer, net_layer in zip(self.brn, self.net):
                x = brn_layer(x, train=brn_train)
                x = net_layer(x)
            q_values = x

        return q_values


class MultiCrossQHead(nn.Module):
    """
    Double Q-network with batch renorm for crossq.
    """

    def __init__(
        self,
        hidden_size,
        action_feature_dim,
        hidden_dims,
        num_q_heads=2,
        output_dim=1,
        train_action_encoder=False,
    ):
        super().__init__()

        self.num_q_heads = num_q_heads
        qs = []
        for q_id in range(self.num_q_heads):
            qs.append(
                CrossQHead(
                    hidden_size,
                    action_feature_dim,
                    hidden_dims,
                    output_dim,
                    train_action_encoder,
                )
            )
        self.qs = nn.ModuleList(qs)

    def forward(
        self,
        state_features,
        action_features,
        next_state_features=None,
        next_action_features=None,
    ):
        """
        Forward pass for both Q-networks.
        if next_state_features and next_action_features are provided, set batch_renorm_train = True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: now and next Q-values, with last dim = num_q_heads
        """
        # process input
        assert (next_state_features is None) == (next_action_features is None), (
            "Either both or none of next_state_features and next_action_features should be provided."
        )
        bs = state_features.shape[0]
        brn_train = next_state_features is not None and next_action_features is not None
        if brn_train:
            cat_state_features = torch.cat([state_features, next_state_features], dim=0)
            cat_action_features = torch.cat(
                [action_features, next_action_features], dim=0
            )
        else:
            cat_state_features = state_features
            cat_action_features = action_features

        q_vs = []
        for qf in self.qs:
            tmp = qf(cat_state_features, cat_action_features, brn_train=brn_train)
            q_vs.append(tmp)

        # the seperated now and next Q-values
        if brn_train:
            now_q_vs = [q_v[:bs] for q_v in q_vs]
            next_q_vs = [q_v[bs:] for q_v in q_vs]
            now_q_vs = torch.cat(now_q_vs, dim=-1)
            next_q_vs = torch.cat(next_q_vs, dim=-1)
        else:
            now_q_vs = q_vs
            next_q_vs = None
            now_q_vs = torch.cat(now_q_vs, dim=-1)

        return now_q_vs, next_q_vs
