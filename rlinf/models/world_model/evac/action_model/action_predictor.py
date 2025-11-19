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


class ActionPredictorMLP(nn.Module):
    """
    predict abs_action[t] based on delta_action[t] and abs_action[t-1]
    Input: delta_action[t] (7) + abs_action[t-1] (8) = 15
    Output: abs_action[t] (8)
    """

    def __init__(
        self,
        input_dim=15,
        output_dim=8,
        hidden_dims=[64, 64],
        dropout=0.0,
        abs_action_mean=None,
        abs_action_std=None,
    ):
        super(ActionPredictorMLP, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Register normalization parameters as buffers
        if abs_action_mean is not None:
            if isinstance(abs_action_mean, torch.Tensor):
                self.register_buffer("abs_action_mean", abs_action_mean)
            else:
                self.register_buffer(
                    "abs_action_mean",
                    torch.tensor(abs_action_mean, dtype=torch.float32),
                )
        else:
            self.register_buffer("abs_action_mean", None)

        if abs_action_std is not None:
            if isinstance(abs_action_std, torch.Tensor):
                self.register_buffer("abs_action_std", abs_action_std)
            else:
                self.register_buffer(
                    "abs_action_std", torch.tensor(abs_action_std, dtype=torch.float32)
                )
        else:
            self.register_buffer("abs_action_std", None)

    def forward(self, delta_action, abs_action_prev):
        """
        Args:
            delta_action: (batch_size, 7) - delta_action[t]
            abs_action_prev: (batch_size, 8) - abs_action[t-1] (should be normalized)
        Returns:
            abs_action_pred: (batch_size, 8) - 预测的 abs_action[t] (normalized)
        """
        x = torch.cat([delta_action, abs_action_prev], dim=1)
        return self.network(x)

    def get_ee_pose(self, delta_action, pre_ee_pose):
        """
        Get next ee_pose with automatic normalize/unnormalize

        Args:
            delta_action: (batch_size, 7) - delta_action[t]
            pre_ee_pose: (batch_size, 8) - previous ee_pose (unnormalized)
        Returns:
            next_ee_pose: (batch_size, 8) - next ee_pose (unnormalized)
        """
        if self.abs_action_mean is None or self.abs_action_std is None:
            raise ValueError(
                "abs_action_mean and abs_action_std must be set for get_ee_pose()"
            )

        # Normalize pre_ee_pose
        pre_ee_pose_normalized = (
            pre_ee_pose - self.abs_action_mean
        ) / self.abs_action_std

        # Forward pass (returns normalized next_ee_pose)
        next_ee_pose_normalized = self.forward(delta_action, pre_ee_pose_normalized)

        # Unnormalize next_ee_pose
        next_ee_pose = (
            next_ee_pose_normalized * self.abs_action_std + self.abs_action_mean
        )

        return next_ee_pose

    def set_normalization_params(self, abs_action_mean, abs_action_std):
        """
        Set normalization parameters after initialization

        Args:
            abs_action_mean: (1, 8) or (8,) tensor or array - mean for normalization
            abs_action_std: (1, 8) or (8,) tensor or array - std for normalization
        """
        if isinstance(abs_action_mean, torch.Tensor):
            self.register_buffer("abs_action_mean", abs_action_mean)
        else:
            self.register_buffer(
                "abs_action_mean", torch.tensor(abs_action_mean, dtype=torch.float32)
            )

        if isinstance(abs_action_std, torch.Tensor):
            self.register_buffer("abs_action_std", abs_action_std)
        else:
            self.register_buffer(
                "abs_action_std", torch.tensor(abs_action_std, dtype=torch.float32)
            )
