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

This module provides a unified interface for all dataset-related classes
used in RLinf training pipelines, including:

- Value datasets (ValueDataset, ValueMixtureDataset, DynamicReturnDataset)
- SFT datasets
- DataLoader implementations
- Data transforms
"""

# Suppress libdav1d/PyAV verbose logging (must be FIRST before any av imports)
import os as _os

_os.environ.setdefault("AV_LOG_FORCE_NOCOLOR", "1")
try:
    import av as _av

    _av.logging.set_level(_av.logging.ERROR)
except ImportError:
    pass

# ============================================================================
# vla_lib datasets (Value learning)
# ============================================================================
# ============================================================================
# DataLoader implementations
# ============================================================================
from rlinf.datasets.dataloaders import (  # noqa: E402
    CFGDataLoaderImpl,
    ValueDataLoaderImpl,
    VlaLibValueModelDataLoaderImpl,
)

# ============================================================================
# Data transforms
# ============================================================================
from rlinf.datasets.transforms import (  # noqa: E402
    TokenizePromptWithGuidance,
)
from rlinf.datasets.vla_lib import (  # noqa: E402
    # RL Dataset
    LeRobotRLDataset,
    # Value transforms
    ReturnDiscretizer,
    ReturnNormalizer,
    # Config
    RLDataConfig,
    # Value Dataset
    ValueDataset,
    # Value Mixture Dataset
    ValueMixtureDataset,
    add_value_tokens_to_tokenizer,
    create_return_discretizer,
    create_rl_config,
    get_all_value_tokens,
    # Value tokens
    get_value_token,
    load_return_range_from_norm_stats,
    parse_value_token,
)

__all__ = [
    # Config
    "RLDataConfig",
    "create_rl_config",
    "load_return_range_from_norm_stats",
    # RL Dataset
    "LeRobotRLDataset",
    # Value Dataset
    "ValueDataset",
    # Value Mixture Dataset
    "ValueMixtureDataset",
    # DataLoaders
    "CFGDataLoaderImpl",
    "VlaLibValueModelDataLoaderImpl",
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
