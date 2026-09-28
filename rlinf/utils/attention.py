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

"""Picking the HuggingFace attention implementation a venv can actually run."""

from __future__ import annotations

import transformers.utils as transformers_utils

from rlinf.utils.logging import get_logger

logger = get_logger()

# Newest first: a venv holds at most one flash-attention variant, because FA2
# and FA4 share the flash_attn package, so at most one of these can be True.
_FLASH_IMPLEMENTATIONS = {
    "flash_attention_4": "is_flash_attn_4_available",
    "flash_attention_3": "is_flash_attn_3_available",
    "flash_attention_2": "is_flash_attn_2_available",
}


def _is_available(implementation: str) -> bool:
    """Whether transformers can run `implementation` in this venv."""
    check = _FLASH_IMPLEMENTATIONS.get(implementation)
    if check is None:
        return True
    # Older transformers releases know fewer flash variants.
    probe = getattr(transformers_utils, check, None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:  # a probe may import the kernels and fail on the platform
        return False


def resolve_attn_implementation(preferred: str = "flash_attention_2") -> str:
    """Return `preferred`, or the closest implementation this venv can run.

    Images built for sm90+ ship FA4 instead of FA2, and platforms without
    flash-attn (Ascend, MUSA) ship neither, so a hard-coded ``flash_attention_2``
    fails there. Fall back to another installed flash variant when there is one,
    and to ``sdpa`` otherwise.
    """
    if _is_available(preferred):
        return preferred
    for implementation in _FLASH_IMPLEMENTATIONS:
        if implementation != preferred and _is_available(implementation):
            logger.warning(
                f"{preferred} is unavailable; using {implementation} instead."
            )
            return implementation
    logger.warning(
        f"{preferred} is unavailable and no flash-attn is installed; using sdpa."
    )
    return "sdpa"
