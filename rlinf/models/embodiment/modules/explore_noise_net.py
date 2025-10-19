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

from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn

from rlinf.utils.logging import get_logger

activation_dict = nn.ModuleDict(
    {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "tanh": nn.Tanh(),
        "mish": nn.Mish(),
        "identity": nn.Identity(),
        "softplus": nn.Softplus(),
        "silu": nn.SiLU(),
    }
)


class ExploreNoiseNet(nn.Module):
    """
    Neural network to generate learnable exploration noise, conditioned on time embeddings and or state embeddings.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        activation_type: str,
        noise_logvar_range: list,  # [min_std, max_std]
        noise_scheduler_type: str,
    ):
        super().__init__()
        self.mlp_logvar = MLP(
            [in_dim] + hidden_dims + [out_dim],
            activation_type=activation_type,
            out_activation_type="Identity",
        )
        self.noise_scheduler_type = noise_scheduler_type
        self.set_noise_range(noise_logvar_range)

    def set_noise_range(self, noise_logvar_range: list):
        self.noise_logvar_range = noise_logvar_range
        noise_logvar_min = self.noise_logvar_range[0]
        noise_logvar_max = self.noise_logvar_range[1]
        self.register_buffer(
            "logvar_min",
            torch.log(torch.tensor(noise_logvar_min**2, dtype=torch.float32)).unsqueeze(
                0
            ),
        )
        self.register_buffer(
            "logvar_max",
            torch.log(torch.tensor(noise_logvar_max**2, dtype=torch.float32)).unsqueeze(
                0
            ),
        )

    def forward(self, noise_feature: torch.Tensor):
        if "const" in self.noise_scheduler_type:  # const or const_schedule_itr
            # pick the lowest noise level when we use constant noise schedulers.
            noise_std = torch.exp(0.5 * self.logvar_min)
        else:
            # use learnable noise level.
            noise_logvar = self.mlp_logvar(noise_feature)
            noise_std = self.post_process(noise_logvar)
        return noise_std

    def post_process(self, noise_logvar):
        """
        input:
            torch.Tensor([B, Ta , Da])
        output:
            torch.Tensor([B, Ta, Da])
        """
        noise_logvar = torch.tanh(noise_logvar)
        noise_logvar = (
            self.logvar_min
            + (self.logvar_max - self.logvar_min) * (noise_logvar + 1) / 2.0
        )
        noise_std = torch.exp(0.5 * noise_logvar)
        return noise_std


class MLP(nn.Module):
    def __init__(
        self,
        dim_list,
        append_dim=0,
        append_layers=None,
        activation_type="tanh",
        out_activation_type="identity",
        use_layernorm=False,
        use_layernorm_final=False,
        dropout=0,
        use_drop_final=False,
        out_bias_init=None,
        verbose=False,
    ):
        super(MLP, self).__init__()

        # Ensure append_layers is always a list to avoid TypeError
        self.append_layers = append_layers if append_layers is not None else []

        # Construct module list
        self.moduleList = nn.ModuleList()
        num_layer = len(dim_list) - 1
        for idx in range(num_layer):
            i_dim = dim_list[idx]
            o_dim = dim_list[idx + 1]
            if append_dim > 0 and idx in self.append_layers:
                i_dim += append_dim
            linear_layer = nn.Linear(i_dim, o_dim)

            # Add module components
            layers = [("linear_1", linear_layer)]
            if use_layernorm and (idx < num_layer - 1 or use_layernorm_final):
                layers.append(("norm_1", nn.LayerNorm(o_dim)))  # type: ignore
            if dropout > 0 and (idx < num_layer - 1 or use_drop_final):
                layers.append(("dropout_1", nn.Dropout(dropout)))  # type: ignore

            # Add activation function
            act = (
                activation_dict[activation_type.lower()]
                if idx != num_layer - 1
                else activation_dict[out_activation_type.lower()]
            )
            layers.append(("act_1", act))  # type: ignore

            # Re-construct module
            module = nn.Sequential(OrderedDict(layers))
            self.moduleList.append(module)
        if verbose:
            get_logger.info(self.moduleList)

        # Initialize the bias of the final linear layer if specified
        if out_bias_init is not None:
            final_linear = self.moduleList[-1][
                0
            ]  # Linear layer is first in the last Sequential # type: ignore
            nn.init.constant_(final_linear.bias, out_bias_init)
            get_logger.info(
                f"Initialized the bias of the final linear layer to {out_bias_init}"
            )

    def forward(self, x, append=None):
        for layer_ind, m in enumerate(self.moduleList):
            if append is not None and layer_ind in self.append_layers:
                x = torch.cat((x, append), dim=-1)
            x = m(x)
        return x
