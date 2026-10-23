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

from rlinf.models.embodiment.openpi.policies import dexjoco_policy


@dataclasses.dataclass(frozen=True)
class LeRobotDexJoCoDataConfig(DataConfigFactory):
    """Use the official DexJoCo LeRobot observation and action layout."""

    dual_arm: bool = False
    image_key: str = "observation.images.front"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_structure = {
            "observation/image": self.image_key,
            "observation/state": "observation.state",
            "actions": "action",
            "prompt": "task",
        }
        if self.dual_arm:
            repack_structure["observation/wrist_image-0"] = (
                "observation.images.wrist_left"
            )
            repack_structure["observation/wrist_image-1"] = (
                "observation.images.wrist_right"
            )
        else:
            repack_structure["observation/wrist_image"] = "observation.images.wrist"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_structure)]
        )

        data_transforms = _transforms.Group(
            inputs=[
                dexjoco_policy.DexJoCoInputs(
                    model_type=model_config.model_type, dual_arm=self.dual_arm
                )
            ],
            outputs=[dexjoco_policy.DexJoCoOutputs(dual_arm=self.dual_arm)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
