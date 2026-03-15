# Copyright 2026 The RLinf Authors.
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

import torch.nn as nn

from .resampler import Resampler, TaskTokenResampler


def build_mlp(in_hidden_size, hidden_size):
    modules = [nn.Linear(in_hidden_size, hidden_size)]
    modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*modules)


def build_expand_mlp(in_hidden_size, hidden_size, out_size):
    modules = [nn.Linear(in_hidden_size, hidden_size)]
    modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, hidden_size))
    modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, out_size))
    return nn.Sequential(*modules)


class DepthHead(nn.Module):
    def __init__(
        self,
        proj_config=None,
        llm_hidden_size=4096,
        use_intermediate_depth=False,
    ):
        super(DepthHead, self).__init__()

        self.projector = Resampler(
            dim_in=llm_hidden_size,
            dim_mid=llm_hidden_size,
            dim_head=proj_config["dim_head"],
            dim_out=proj_config["dim_out"],
            num_layers=proj_config["num_layers"],
            num_heads=proj_config["num_heads"],
            num_queries=proj_config["num_backbone_tokens"],
            ff_mult=proj_config["ff_mult"],
        )

    def forward(self, llm_feats):
        queries = self.projector(llm_feats)
        return queries


class TaskTokenDepthHead(nn.Module):
    def __init__(
        self,
        proj_config=None,
        llm_hidden_size=4096,
        use_intermediate_depth=False,
    ):
        super(TaskTokenDepthHead, self).__init__()

        self.projector = TaskTokenResampler(
            dim_in=llm_hidden_size,
            dim_mid=llm_hidden_size,
            dim_head=proj_config["dim_head"],
            dim_out=proj_config["dim_out"],
            num_layers=proj_config["num_layers"],
            num_heads=proj_config["num_heads"],
            num_queries=proj_config["num_backbone_tokens"],
            ff_mult=proj_config["ff_mult"],
        )

    def forward(self, llm_feats, queries):
        queries = self.projector(llm_feats, queries)
        return queries
