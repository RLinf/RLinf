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
# openpi model configs
import dataclasses
import difflib
from typing import Optional

import openpi.models.pi0_config as pi0_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    TrainConfig,
)

from rlinf.models.embodiment.openpi.dataconfig.calvin_dataconfig import (
    LeRobotCalvinDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.franka_dataconfig import (
    CustomDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.libero_dataconfig import (
    LeRobotLiberoDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.maniskill_dataconfig import (
    LeRobotManiSkillDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.metaworld_dataconfig import (
    LeRobotMetaworldDataConfig,
)

_CONFIGS = [
    TrainConfig(
        name="pi0_maniskill",
        model=pi0_config.Pi0Config(),
        data=LeRobotManiSkillDataConfig(
            repo_id="physical-intelligence/maniskill",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base"),
            extra_delta_transform=True,
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
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
    ),
    TrainConfig(
        name="pi0_calvin",
        model=pi0_config.Pi0Config(action_horizon=5),
        data=LeRobotCalvinDataConfig(
            repo_id="InternRobotics/InternData-Calvin_ABC",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_calvin/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi0_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi05_calvin",
        model=pi0_config.Pi0Config(
            pi05=True, action_horizon=5, discrete_state_input=False
        ),
        data=LeRobotCalvinDataConfig(
            repo_id="InternRobotics/InternData-Calvin_ABC",
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_calvin/assets"),
            extra_delta_transform=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "checkpoints/jax/pi05_base/params"
        ),
        pytorch_weight_path="checkpoints/torch/pi05_base",
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_custom",
        model=pi0_config.Pi0Config(),
        data=CustomDataConfig(
            repo_id="physical-intelligence/custom_dataset",
            base_config=DataConfig(
                prompt_from_task=True
            ),  # we need language instruction
            assets=AssetsConfig(assets_dir="checkpoints/torch/pi0_base/assets"),
            raw_action_is_delta=True,  # True for delta action, False for abs_action
            action_train_with_rotation_6d=False,
        ),
        pytorch_weight_path="checkpoints/torch/pi0_base",
    ),
]


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def _override_with_model_path(config: TrainConfig, model_path: str) -> TrainConfig:
    """Return a copy of the config with assets/weight paths set from model_path."""
    data_config = config.data
    if (
        dataclasses.is_dataclass(data_config)
        and hasattr(data_config, "assets")
        and dataclasses.is_dataclass(data_config.assets)
    ):
        data_config = dataclasses.replace(
            data_config,
            assets=dataclasses.replace(data_config.assets, assets_dir=model_path),
        )

    replace_kwargs = {
        "data": data_config,
        "pytorch_weight_path": model_path,
    }
    if dataclasses.is_dataclass(config) and any(
        field.name == "assets_dirs" for field in dataclasses.fields(config)
    ):
        replace_kwargs["assets_dirs"] = model_path

    return dataclasses.replace(config, **replace_kwargs)


def get_openpi_config(
    config_name: str, model_path: Optional[str] = None
) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    config = _CONFIGS_DICT[config_name]
    if model_path is not None:
        config = _override_with_model_path(config, model_path)

    return config
