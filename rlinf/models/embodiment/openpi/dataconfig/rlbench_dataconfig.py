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
"""
RLBench data config for OpenPI. Compatible with pi05_rlbench checkpoint
(LeRobot format, asset_id=rlbench18). Action normalization flow matches
Metaworld: same data_transforms + model_transforms, openpi Normalize/Unnormalize only.
"""
import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from rlinf.models.embodiment.openpi.policies import rlbench_policy
from typing_extensions import override


@dataclasses.dataclass(frozen=True)
class LeRobotRLBenchDataConfig(DataConfigFactory):
    """
    Config for RLBench 18-tasks dataset (LeRobot format).
    Uses convert_rlbench_to_lerobot.py output with: image, overhead_image,
    wrist_image, state (7D), actions (7D).
    asset_id rlbench18 matches checkpoint norm_stats path.
    """

    extra_delta_transform: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/overhead_image": "overhead_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        num_views = getattr(model_config, "num_images_in_input", 3)
        data_transforms = _transforms.Group(
            inputs=[
                rlbench_policy.RLBenchInputs(
                    model_type=model_config.model_type,
                    num_views=num_views,
                )
            ],
            outputs=[rlbench_policy.RLBenchOutputs()],
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
        )
