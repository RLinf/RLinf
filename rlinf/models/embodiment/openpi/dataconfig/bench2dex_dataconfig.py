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
import openpi.transforms as transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

from rlinf.models.embodiment.openpi.policies import bench2dex_policy


@dataclasses.dataclass(frozen=True)
class LeRobotBench2DexDataConfig(DataConfigFactory):
    """Read converted Bench2Dex RGB and active-joint demonstrations."""

    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        cameras = bench2dex_policy.CAMERA_NAMES
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": f"observation.images.{cameras[0]}",
                        "observation/extra_view_image": [
                            f"observation.images.{cam}" for cam in cameras[1:]
                        ],
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=transforms.Group(
                inputs=[bench2dex_policy.Bench2DexInputs()],
                outputs=[bench2dex_policy.Bench2DexOutputs()],
            ),
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )
