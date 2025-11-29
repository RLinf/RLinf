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
import difflib

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    TrainConfig,
)

from rlinf.models.embodiment.openpi.dataconfig.behavior_dataconfig import (
    LeRobotBehaviorDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.libero_dataconfig import (
    LeRobotLiberoDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.metaworld_dataconfig import (
    LeRobotMetaworldDataConfig,
)

_CONFIGS = [
    TrainConfig(
        name="pi0_libero",
        model=pi0_config.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_libero/assets"),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_metaworld",
        model=pi0_config.Pi0Config(action_horizon=5),
        data=LeRobotMetaworldDataConfig(
            repo_id="lerobot/metaworld_mt50",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_metaworld/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_behavior",
        model=pi0_config.Pi0Config(),
        data=LeRobotBehaviorDataConfig(
            repo_id="physical-intelligence/behavior",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_behavior/assets"),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_metaworld",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=5, discrete_state_input=False
        ),
        data=LeRobotMetaworldDataConfig(
            repo_id="lerobot/metaworld_mt50",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_metaworld/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_behavior",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=10, discrete_state_input=False
        ),
        data=LeRobotBehaviorDataConfig(
            repo_id="physical-intelligence/behavior",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_behavior/assets"),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=30_000,
    ),
]


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def get_openpi_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
