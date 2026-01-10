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
import dataclasses
import logging
import pathlib
from typing import Optional

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    ModelTransformFactory,
)
from typing_extensions import override

from rlinf.models.embodiment.openpi.dataconfig.libero_dataconfig import (
    LeRobotLiberoDataConfig,
)
from rlinf.models.embodiment.openpi.dataconfig.maniskill_dataconfig import (
    LeRobotManiSkillDataConfig,
)
from rlinf.models.embodiment.openpi.policies import libero_policy, maniskill_policy


@dataclasses.dataclass(frozen=True)
class CombinedDataConfig(DataConfig):
    """
    Extended DataConfig that supports combined datasets.
    This allows storing multiple sub-configs for mixed dataset training.
    """
    combined_datasets: Optional[list[DataConfig]] = None


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoManiSkillDataConfig(DataConfigFactory):
    """
    This config combines Libero and ManiSkill datasets for training.
    Libero will only use base image (front view) to align with ManiSkill format.
    """
    repo_id: str = "combined_libero_maniskill"
    # First dataset config (Libero) - will use base image only
    libero_config: LeRobotLiberoDataConfig = dataclasses.field(default_factory=lambda: LeRobotLiberoDataConfig(
        repo_id="physical-intelligence/libero",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir="checkpoints/torch/pi0_libero/assets"
        ),
        extra_delta_transform=False,
    ))
    
    # Second dataset config (ManiSkill)
    maniskill_config: "LeRobotManiSkillDataConfig" = dataclasses.field(default_factory=lambda: LeRobotManiSkillDataConfig(
        repo_id="physical-intelligence/maniskill",
        base_config=DataConfig(prompt_from_task=True),
        assets=AssetsConfig(
            assets_dir="checkpoints/torch/pi0_maniskill/assets"
        ),
        extra_delta_transform=True,
    ))
    
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CombinedDataConfig:
        # Create base libero config first
        # Use the assets_dir from libero_config.assets if available, otherwise use assets_dirs
        libero_assets_dir = pathlib.Path(
            self.libero_config.assets.assets_dir
            if hasattr(self.libero_config, "assets") and self.libero_config.assets.assets_dir
            else assets_dirs
        )
        libero_base_config = self.libero_config.create_base_config(libero_assets_dir, model_config)
        
        # Override data_transforms to use LiberoInputsBaseOnly (only base image, no wrist)
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",  # Still repack it, but won't be used
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        
        # Use LiberoInputs which will handle missing wrist_image by using zeros
        # The image_mask will be set to False for left_wrist_0_rgb when wrist_image is missing
        libero_data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
        
        if self.libero_config.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            libero_data_transforms = libero_data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        
        libero_model_transforms = ModelTransformFactory()(model_config)
        
        libero_data_config = dataclasses.replace(
            libero_base_config,
            repack_transforms=repack_transform,
            data_transforms=libero_data_transforms,
            model_transforms=libero_model_transforms,
        )
        
        # Create configs for ManiSkill dataset
        # Each sub-config will load its own norm_stats from its assets directory
        # Use the assets_dir from maniskill_config.assets if available, otherwise use assets_dirs
        maniskill_assets_dir = pathlib.Path(
            self.maniskill_config.assets.assets_dir
            if hasattr(self.maniskill_config, "assets") and self.maniskill_config.assets.assets_dir
            else assets_dirs
        )
        maniskill_data_config = self.maniskill_config.create(maniskill_assets_dir, model_config)
        
        # Verify that each sub-config has its own norm_stats loaded
        if libero_data_config.norm_stats is None:
            logging.warning("Libero norm_stats not loaded. Make sure to run compute_norm_stats for libero dataset.")
        else:
            logging.info(f"Libero norm_stats loaded with keys: {list(libero_data_config.norm_stats.keys())}")
        
        if maniskill_data_config.norm_stats is None:
            logging.warning("ManiSkill norm_stats not loaded. Make sure to run compute_norm_stats for maniskill dataset.")
        else:
            logging.info(f"ManiSkill norm_stats loaded with keys: {list(maniskill_data_config.norm_stats.keys())}")
        
        # Return a combined config with both sub-configs
        # The data loader will handle creating the combined dataset
        # Each sub-dataset will use its own norm_stats during transform_dataset
        # Convert libero_data_config to CombinedDataConfig
        # Get all fields from DataConfig and libero_data_config
        data_config_fields = {field.name for field in dataclasses.fields(DataConfig)}
        libero_config_dict = {
            field.name: getattr(libero_data_config, field.name)
            for field in dataclasses.fields(libero_data_config)
            if field.name in data_config_fields
        }
        # Override repo_id with the combined dataset identifier
        libero_config_dict["repo_id"] = "combined_libero_maniskill"
        # Create CombinedDataConfig with all fields from libero_data_config plus combined_datasets
        return CombinedDataConfig(
            **libero_config_dict,
            combined_datasets=[libero_data_config, maniskill_data_config],
        )
