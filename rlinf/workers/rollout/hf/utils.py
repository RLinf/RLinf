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

from typing import Union

import torch

from rlinf.utils.nested_dict_process import copy_dict_tensor


def init_real_obs(next_extracted_obs: Union[torch.Tensor, dict]):
    # Copy the next-extracted-obs
    if isinstance(next_extracted_obs, torch.Tensor):
        real_next_extracted_obs = next_extracted_obs.clone()
    elif isinstance(next_extracted_obs, dict):
        real_next_extracted_obs = copy_dict_tensor(next_extracted_obs)
    else:
        raise NotImplementedError
    return real_next_extracted_obs
