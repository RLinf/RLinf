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
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import calvin_policy


@dataclasses.dataclass(frozen=True)
class LeRobotCalvinDataConfig(DataConfigFactory):
    extra_delta_transform: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "video.image_base",
                        "observation/wrist_image": "video.image_wrist",
                        "observation/state_ee_pos": "state.ee_pos",
                        "observation/state_ee_rot": "state.ee_rot",
                        "observation/state_gripper": "state.gripper",
                        "actions/delta_ee_pos": "action.delta_ee_pos",
                        "actions/delta_ee_rot": "action.delta_ee_rot",
                        "actions/gripper": "action.gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[calvin_policy.CalvinInputs(model_type=model_config.model_type)],
            outputs=[calvin_policy.CalvinOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=(
                "action.delta_ee_pos",
                "action.delta_ee_rot",
                "action.gripper",
            ),
        )
