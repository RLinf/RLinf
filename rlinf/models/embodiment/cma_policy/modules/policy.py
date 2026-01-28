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

import abc
from typing import Any

import torch.nn as nn
from habitat_baselines.utils.common import CategoricalNet

from rlinf.models.embodiment.cma_policy.modules.utils import CustomFixedCategorical


class CMABasePolicy(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, net, dim_actions):
        """Defines an imitation learning policy as having functions act() and
        build_distribution().
        """
        super(CMABasePolicy, self).__init__()
        self.net = net
        self.dim_actions = dim_actions

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_states = self.net(observations, rnn_states, prev_actions, masks)
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        return action, rnn_states

    def get_value(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def evaluate_actions(self, *args: Any, **kwargs: Any):
        raise NotImplementedError

    def build_distribution(
        self, observations, rnn_states, prev_actions, masks
    ) -> CustomFixedCategorical:
        features, rnn_states = self.net(observations, rnn_states, prev_actions, masks)
        return self.action_distribution(features)
