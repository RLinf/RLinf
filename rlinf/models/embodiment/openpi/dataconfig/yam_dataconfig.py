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

"""OpenPI data configuration for dual-YAM LeRobot demonstrations."""

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import yam_policy


@dataclasses.dataclass(frozen=True)
class LeRobotYamDataConfig(DataConfigFactory):
    """Use joint deltas for training and recover absolute targets for execution.

    Only the twelve revolute joints use deltas; both normalized gripper
    openings stay absolute, matching pi05_yam_joint checkpoints used by RPent.
    """

    default_prompt: str | None = None

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
        delta_mask = np.array([True] * 6 + [False] + [True] * 6 + [False])
        data_transforms = _transforms.Group(
            inputs=[yam_policy.YamInputs()],
            outputs=[yam_policy.YamOutputs()],
        ).push(
            inputs=[_transforms.DeltaActions(delta_mask)],
            outputs=[_transforms.AbsoluteActions(delta_mask)],
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
