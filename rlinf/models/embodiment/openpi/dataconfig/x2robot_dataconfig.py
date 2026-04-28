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

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import x2robot_policy


@dataclasses.dataclass(frozen=True)
class LeRobotX2RobotDataConfig(DataConfigFactory):
    action_dim: int = 14

    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "left_wrist_view": "left_wrist_view",
                            "face_view": "face_view",
                            "right_wrist_view": "right_wrist_view",
                        },
                        "state": "state",
                        "actions": "actions",
                        "actions_is_pad": "actions_is_pad",
                        "prompt": "task",
                    }
                )
            ]
        )
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[
                x2robot_policy.X2RobotInputs(
                    action_dim=model_config.action_dim,
                )
            ],
            outputs=[x2robot_policy.X2RobotOutputs(action_dim=self.action_dim)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
