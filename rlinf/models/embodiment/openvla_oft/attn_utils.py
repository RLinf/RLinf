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

from collections.abc import Mapping
from typing import Any

import torch


def is_musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def resolve_attn_implementation(cfg: Mapping[str, Any]) -> str:
    """Resolve attention implementation for the current device/runtime."""
    attn_implementation = cfg.get("attn_implementation", "flash_attention_2")
    force_musa_eager_attn_implementation = bool(
        cfg.get("force_musa_eager_attn_implementation", True)
    )

    if is_musa_available() and force_musa_eager_attn_implementation:
        return "eager"

    return attn_implementation
