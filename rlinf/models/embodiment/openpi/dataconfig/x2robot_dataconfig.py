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
    mode: str = "s2s"
    action_dim: int = 14
    use_delta_actions: bool = False
    only_right_obs: bool = False
    random_pos_offset: float = 0.0

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
        if self.mode != "s2s":
            raise ValueError(
                "RLinf Turtle2/X2Robot OpenPI integration currently supports only "
                f"mode='s2s', got {self.mode!r}."
            )

        data_transforms = _transforms.Group(
            inputs=[
                x2robot_policy.X2RobotInputs(
                    action_dim=model_config.action_dim,
                    only_right_obs=self.only_right_obs,
                    random_pos_offset=self.random_pos_offset,
                )
            ],
            outputs=[x2robot_policy.X2RobotOutputs(action_dim=self.action_dim)],
        )

        if self.use_delta_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
