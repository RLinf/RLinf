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


class BatchRenorm(nn.Module):
    """
    Batch Renormalization: Towards Reducing Internal Covariate Shift in Deep Learning.
    https://arxiv.org/abs/1702.03275
    """

    def __init__(self, num_features, eps=1e-3, momentum=0.999, r_max=3.0, d_max=5.0):
        """
        Initializes the BatchRenorm module.

        Args:
            num_features (int): The number of features in the input tensor.
            eps (float): A small value added to the variance for numerical stability.
            momentum (float): The momentum for updating the running statistics.
            r_max (float): The maximum value for the correction factor 'r'.
            d_max (float): The maximum value for the correction factor 'd'.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # Learnable parameters for affine transformation
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        # Buffers for running statistics and renormalization limits
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("r_max", torch.tensor(r_max))
        self.register_buffer("d_max", torch.tensor(d_max))
        self.register_buffer("steps", torch.tensor(0, dtype=torch.long))

    def forward(self, x, train=False):
        """
        Performs the forward pass of the BatchRenorm module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized output tensor.
        """
        # Ensure input is at least 2D
        if x.dim() < 2:
            raise ValueError(f"Expected 2D or higher input (got {x.dim()}D input)")

        # Reshape input for feature dimension
        if x.dim() > 2:
            x = x.view(x.size(0), self.num_features, -1)
            x = x.permute(0, 2, 1).contiguous()
            x = x.view(-1, self.num_features)

        if train:
            # Calculate batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            batch_std = torch.sqrt(batch_var + self.eps)

            # Stop gradients for renormalization factors
            with torch.no_grad():
                running_std = torch.sqrt(self.running_var + self.eps)

                # Calculate r and d
                r = batch_std / running_std
                d = (batch_mean - self.running_mean) / running_std

                # Clip r and d
                r = torch.clamp(r, 1.0 / self.r_max, self.r_max)
                d = torch.clamp(d, -self.d_max, self.d_max)

            # Renormalize the input
            x_normalized = (x - batch_mean) / batch_std * r + d

            # Update running statistics (detached to avoid graph reuse)
            with torch.no_grad():
                self.running_mean = (
                    self.momentum * self.running_mean
                    + (1 - self.momentum) * batch_mean.detach()
                )
                self.running_var = (
                    self.momentum * self.running_var
                    + (1 - self.momentum) * batch_var.detach()
                )
                self.steps += 1
        else:
            # Use running statistics for normalization in evaluation mode
            x_normalized = (x - self.running_mean) / torch.sqrt(
                self.running_var + self.eps
            )

        # Apply affine transformation
        out = self.weight * x_normalized + self.bias

        return out.view(x.shape) if x.dim() > 2 else out


class placeholder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, train=False):
        return x


def make_batchrenorm(in_channels, mlp_channels, last_act=True):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(BatchRenorm(c_in))
        if last_act or idx < len(mlp_channels) - 1:
            module_list.append(placeholder())  # Placeholder for activation if needed
        c_in = c_out
    return module_list
