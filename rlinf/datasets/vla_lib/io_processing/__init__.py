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
I/O Processing for RL Datasets.

This module provides robot-specific transforms for RL value learning.
Extends the VLA io_processing with RL-specific output handling.
"""

from rlinf.datasets.vla_lib.lerobot_datasets.io_processing.libero import (
    LiberoInputs,
    LiberoOutputs,
)

from .value_tokens import (
    DEFAULT_NUM_VALUE_BINS,
    add_value_tokens_to_tokenizer,
    get_all_value_tokens,
    get_value_token,
    parse_value_token,
)

# RL-specific value transforms
from .value_transforms import (
    ReturnDiscretizer,
    ReturnNormalizer,
    create_return_discretizer,
)

__all__ = [
    # VLA transforms
    "LiberoInputs",
    "LiberoOutputs",
    # RL value transforms
    "ReturnDiscretizer",
    "ReturnNormalizer",
    "create_return_discretizer",
    # Value tokens
    "get_value_token",
    "get_all_value_tokens",
    "parse_value_token",
    "add_value_tokens_to_tokenizer",
    "DEFAULT_NUM_VALUE_BINS",
]
