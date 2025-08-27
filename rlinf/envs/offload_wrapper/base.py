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
from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG


def recursive_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to_device(elem, device) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_device(elem, device) for elem in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


def get_batch_rng_state(batched_rng):
    state = {
        "rngs": batched_rng.rngs,
    }
    return state


def set_batch_rng_state(state: dict):
    return BatchedRNG.from_rngs(state["rngs"])


class EnvOffloadMixin:
    def get_state(self) -> bytes:
        pass

    def load_state(self, state: bytes):
        pass
