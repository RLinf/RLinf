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

import torch.nn as nn


class BasePolicy(nn.Module):
    def preprocess_env_obs(self, env_obs):
        return env_obs

    def forward(self, forward_type="default_forward", **kwargs):
        if forward_type == "default_forward":
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sac_forward(self, **kwargs):
        raise NotImplementedError

    def sac_q_forward(self, **kwargs):
        raise NotImplementedError

    def crossq_forward(self, **kwargs):
        raise NotImplementedError

    def crossq_q_forward(self, **kwargs):
        raise NotImplementedError

    def default_forward(self, **kwargs):
        raise NotImplementedError
