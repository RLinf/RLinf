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

"""
RLinf Datasets Module.

Provides dataset classes, dataloader wrappers, and transforms for
value learning, CFG training, and SFT pipelines.
"""

# Suppress libdav1d/PyAV verbose logging (must be FIRST before any av imports)
import os as _os

_os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
try:
    import av as _av

    _av.logging.set_level(_av.logging.ERROR)
except ImportError:
    pass

import dataclasses
from typing import Any, Iterator

# ============================================================================
# Config
# ============================================================================
from rlinf.datasets.config import (  # noqa: E402
    RLDataConfig,
    create_rl_config,
    load_return_range_from_norm_stats,
)

# ============================================================================
# Datasets
# ============================================================================
from rlinf.datasets.mixture_datasets import (  # noqa: E402
    AdvantageMixtureDataset,
    ValueMixtureDataset,
)
from rlinf.datasets.rl_dataset import LeRobotRLDataset  # noqa: E402
from rlinf.datasets.value_dataset import ValueDataset  # noqa: E402

# ============================================================================
# Value transforms & tokens
# ============================================================================
from rlinf.datasets.value_transforms import (  # noqa: E402
    ReturnDiscretizer,
    ReturnNormalizer,
    add_value_tokens_to_tokenizer,
    create_return_discretizer,
    get_all_value_tokens,
    get_value_token,
    parse_value_token,
)


# ============================================================================
# DataLoader implementations (inlined)
# ============================================================================
class ValueDataLoaderImpl:
    """Lightweight wrapper that yields batches and exposes data_config().

    Used by FSDPValueSftWorker.
    """

    def __init__(self, data_config: dict, data_loader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> dict:
        return self._data_config

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self._data_loader.sampler, "set_epoch"):
            self._data_loader.sampler.set_epoch(epoch)
        if hasattr(self._data_loader.dataset, "set_epoch"):
            self._data_loader.dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._data_loader


# ============================================================================
# Transforms (inlined)
# ============================================================================
@dataclasses.dataclass(frozen=True)
class TokenizePromptWithGuidance:
    """Tokenize both original prompt and guidance prompts for CFG models.

    Generates positive and negative guidance prompts:
    - positive: "[POSITIVE][POSITIVE]\\nTask: {prompt}"
    - negative: "[NEGATIVE][NEGATIVE]\\nTask: {prompt}"
    """

    tokenizer: Any  # openpi.models.tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False

    def __call__(self, data: dict) -> dict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        tokens, token_masks = self.tokenizer.tokenize(prompt, state)

        positive_prompt = f"[POSITIVE][POSITIVE]\nTask: {prompt}"
        negative_prompt = f"[NEGATIVE][NEGATIVE]\nTask: {prompt}"

        positive_tokens, positive_masks = self.tokenizer.tokenize(
            positive_prompt, state
        )
        negative_tokens, negative_masks = self.tokenizer.tokenize(
            negative_prompt, state
        )

        return {
            **data,
            "tokenized_prompt": tokens,
            "tokenized_prompt_mask": token_masks,
            "tokenized_positive_guidance_prompt": positive_tokens,
            "tokenized_positive_guidance_prompt_mask": positive_masks,
            "tokenized_negative_guidance_prompt": negative_tokens,
            "tokenized_negative_guidance_prompt_mask": negative_masks,
        }


__all__ = [
    # Config
    "RLDataConfig",
    "create_rl_config",
    "load_return_range_from_norm_stats",
    # RL Dataset
    "LeRobotRLDataset",
    # Value Dataset
    "ValueDataset",
    # Mixture Datasets
    "AdvantageMixtureDataset",
    "ValueMixtureDataset",
    # DataLoaders
    "ValueDataLoaderImpl",
    # Transforms
    "TokenizePromptWithGuidance",
    # Value transforms
    "ReturnDiscretizer",
    "ReturnNormalizer",
    "create_return_discretizer",
    # Value tokens
    "get_value_token",
    "get_all_value_tokens",
    "parse_value_token",
    "add_value_tokens_to_tokenizer",
]
