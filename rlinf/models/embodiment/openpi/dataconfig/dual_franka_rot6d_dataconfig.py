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
"""Data config for dual-Franka rot6d SFT — body-frame SE(3) delta on rot6d
(component-wise subtraction would break on rotations)."""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import dual_franka_rot6d_policy
from rlinf.models.embodiment.openpi.transforms import (
    DUAL_ARM_ROT6D_LAYOUT,
    RigidBodyAbsoluteActions,
    RigidBodyDeltaActions,
)


@dataclasses.dataclass(frozen=True)
class DualFrankaRot6dDataConfig(DataConfigFactory):
    default_prompt: str | None = None

    # SE(3) delta at train, absolute recovery at inference. pi0/pi05 trains
    # on deltas, so default True.
    extra_delta_transform: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/extra_view_image-0": "extra_view_image-0",
                        "observation/extra_view_image-1": "extra_view_image-1",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                dual_franka_rot6d_policy.DualFrankaRot6dInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[dual_franka_rot6d_policy.DualFrankaRot6dOutputs()],
        )

        if self.extra_delta_transform:
            data_transforms = data_transforms.push(
                inputs=[RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)],
                outputs=[RigidBodyAbsoluteActions(DUAL_ARM_ROT6D_LAYOUT)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
        )
