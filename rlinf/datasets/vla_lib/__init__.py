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

"""
RL Datasets for Value Learning and CFG-RL Training.

Extends VLA datasets with:
1. History observations (past N timesteps)
2. Action/reward chunks (future H timesteps)
3. Next observation for bootstrapping
4. Precomputed returns
5. Return discretization for language model value prediction
6. Value dataset for VLM-based return prediction training
7. Value mixture dataset for training on multiple datasets
8. Dynamic return dataset for on-the-fly return computation
9. Advantage mixture dataset for multi-dataset CFG-RL training
"""

from .advantage_mixture_dataset import AdvantageMixtureDataset
from .config import RLDataConfig, create_rl_config, load_return_range_from_norm_stats
from .io_processing import (
    ReturnDiscretizer,
    ReturnNormalizer,
    add_value_tokens_to_tokenizer,
    create_return_discretizer,
    get_all_value_tokens,
    get_value_token,
    parse_value_token,
)
from .rl_dataset import LeRobotRLDataset, create_rl_dataset
from .value_dataset import ValueDataset, create_value_dataset
from .value_mixture_dataset import (
    ValueMixtureDataset,
    create_single_value_dataset_from_config,
    create_value_mixture_dataset,
    create_value_mixture_dataset_from_config,
)

__all__ = [
    # RL Dataset
    "create_rl_dataset",
    "LeRobotRLDataset",
    # Value Dataset
    "create_value_dataset",
    "ValueDataset",
    # Value Mixture Dataset
    "create_value_mixture_dataset",
    "create_value_mixture_dataset_from_config",
    "create_single_value_dataset_from_config",
    "ValueMixtureDataset",
    # Advantage Mixture Dataset (for CFG-RL)
    "AdvantageMixtureDataset",
    # Config
    "RLDataConfig",
    "create_rl_config",
    "load_return_range_from_norm_stats",
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
