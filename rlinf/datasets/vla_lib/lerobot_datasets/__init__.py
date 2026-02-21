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
LeRobot dataset configurations and transforms.

This module provides dataset configuration classes and transforms that work with
LeRobot format datasets for VLA training.
"""

from .config import (
    DATASET_CONFIGS,
    AlohaDataConfig,
    DataConfig,
    DataConfigFactory,
    DroidDataConfig,
    Franka3CamDataConfig,
    FrankaDataConfig,
    LiberoDataConfig,
    LiberoV2DataConfig,
    create_data_config_factory,
    create_data_config_factory_from_dict,
    detect_robot_type,
    get_dataset_config,
)
from .lerobot_dataset import (
    LeRobotPyTorchDataset,
    TransformedDataset,
    create_lerobot_dataset,
    vla_data_collator,
)
from .normalize import (
    NormStats,
    RunningStats,
    get_mixture_norm_stats_path,
    load_mixture_norm_stats,
    load_stats,
    save_stats,
    validate_mixture_norm_stats,
)
from .transforms import (
    AbsoluteActions,
    CompositeTransform,
    DataTransformFn,
    DeltaActions,
    FASTTokenizerTransform,
    Group,
    InjectDefaultPrompt,
    LiberoEcotBBoxReasoningTransform,
    LiberoEcotPointingReasoningTransform,
    LiberoEcotSubtaskReasoningTransform,
    Normalize,
    PadStatesAndActions,
    PromptFromLeRobotTask,
    RepackTransform,
    ResizeImages,
    SubtaskReasoningTransform,
    Unnormalize,
    compose,
    load_subtask_descriptions,
    load_task_descriptions,
    load_task_to_key_mapping,
    make_bool_mask,
    pad_to_dim,
)

__all__ = [
    # Config
    "DataConfig",
    "DataConfigFactory",
    "LiberoDataConfig",
    "LiberoV2DataConfig",
    "DroidDataConfig",
    "AlohaDataConfig",
    "FrankaDataConfig",
    "Franka3CamDataConfig",
    "DATASET_CONFIGS",
    "get_dataset_config",
    "detect_robot_type",
    "create_data_config_factory",
    "create_data_config_factory_from_dict",
    # Dataset
    "TransformedDataset",
    "LeRobotPyTorchDataset",
    "create_lerobot_dataset",
    "vla_data_collator",
    # Transforms
    "DataTransformFn",
    "Group",
    "CompositeTransform",
    "compose",
    "RepackTransform",
    "InjectDefaultPrompt",
    "ResizeImages",
    "DeltaActions",
    "AbsoluteActions",
    "PromptFromLeRobotTask",
    "Normalize",
    "Unnormalize",
    "PadStatesAndActions",
    "pad_to_dim",
    "make_bool_mask",
    "load_task_descriptions",
    "load_subtask_descriptions",
    "load_task_to_key_mapping",
    "FASTTokenizerTransform",
    "SubtaskReasoningTransform",
    "LiberoEcotSubtaskReasoningTransform",
    "LiberoEcotBBoxReasoningTransform",
    "LiberoEcotPointingReasoningTransform",
    # Normalize
    "NormStats",
    "RunningStats",
    "save_stats",
    "load_stats",
    "get_mixture_norm_stats_path",
    "validate_mixture_norm_stats",
    "load_mixture_norm_stats",
]
