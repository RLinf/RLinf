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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_act_func(activation):
    if activation.lower() == "relu":
        act = nn.ReLU
    elif activation.lower() == "gelu":
        act = nn.GELU
    elif activation.lower() == "tanh":
        act = nn.Tanh
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return act


def make_mlp(
    in_channels, mlp_channels, act_builder=nn.ReLU, last_act=True, use_layer_norm=False
):
    c_in = in_channels
    module_list = []
    for idx, c_out in enumerate(mlp_channels):
        module_list.append(nn.Linear(c_in, c_out))
        if last_act or idx < len(mlp_channels) - 1:
            if use_layer_norm:
                module_list.append(nn.LayerNorm(c_out))
            module_list.append(act_builder())
        c_in = c_out
    return module_list


def init_mlp_weights(mlp, nonlinearity):
    for m in mlp:
        if isinstance(m, nn.Linear):
            if m is mlp[-1]:
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity=nonlinearity
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
