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


def allow_pci_render_backend() -> None:
    """Let ManiSkill accept a full ``pci:<domain>:<bus>:<slot>.<func>`` render backend.

    ManiSkill documents ``pci:...`` as the way to pick a renderer on machines
    without CUDA, but ``parse_backend_device_id`` splits the string on every
    colon and unpacks exactly two values, so a real PCI address raises
    ``ValueError`` before SAPIEN sees it. Keep such a backend intact and let
    SAPIEN resolve it; everything else keeps ManiSkill's own parsing.

    Idempotent, and a no-op for a ManiSkill that is absent or parses backend
    strings some other way.
    """
    try:
        from mani_skill.envs.utils.system import backend
    except ImportError:
        return

    original_parse = getattr(backend, "parse_backend_device_id", None)
    if original_parse is None or getattr(original_parse, "_rlinf_pci_patched", False):
        return

    def parse_backend_device_id(device_backend):
        if isinstance(device_backend, str) and device_backend.startswith("pci:"):
            return device_backend, None
        return original_parse(device_backend)

    parse_backend_device_id._rlinf_pci_patched = True
    backend.parse_backend_device_id = parse_backend_device_id


def recursive_to_own(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone() if obj.is_shared() else obj
    elif isinstance(obj, list):
        return [recursive_to_own(elem) for elem in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to_own(elem) for elem in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to_own(v) for k, v in obj.items()}
    else:
        return obj


def get_batch_rng_state(batched_rng):
    state = {
        "rngs": batched_rng.rngs,
    }
    return state


def set_batch_rng_state(state: dict):
    from mani_skill.envs.utils.randomization.batched_rng import BatchedRNG

    return BatchedRNG.from_rngs(state["rngs"])
