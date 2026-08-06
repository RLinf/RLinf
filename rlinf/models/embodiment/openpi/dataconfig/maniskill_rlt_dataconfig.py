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

from rlinf.models.embodiment.openpi.policies import maniskill_policy


@dataclasses.dataclass(frozen=True)
class LeRobotRLTManiSkillJointDataConfig(DataConfigFactory):
    """DataConfig for ManiSkill joint-space RLT LeRobot datasets."""

    extra_delta_transform: bool = False
    image_key: str = "image"
    wrist_image_key: str = "wrist_image"
    extra_view_image_key: str | None = None
    state_key: str = "state"
    action_key: str = "actions"
    task_key: str | None = None
    prompt_key: str | None = None
    default_prompt: str | None = None
    output_action_dim: int = 8

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_structure = {
            "observation/image": self.image_key,
            "observation/wrist_image": self.wrist_image_key,
            "observation/state": self.state_key,
            "actions": self.action_key,
        }
        if self.prompt_key is not None:
            repack_structure["prompt"] = self.prompt_key
        if self.task_key is not None:
            repack_structure["task"] = self.task_key
        if self.extra_view_image_key is not None:
            repack_structure["observation/extra_view_image"] = self.extra_view_image_key

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_structure)]
        )

        data_transforms = _transforms.Group(
            inputs=[
                maniskill_policy.ManiSkillInputs(
                    model_type=model_config.model_type,
                    use_wrist_image=True,
                    default_prompt=self.default_prompt,
                )
            ],
            outputs=[
                maniskill_policy.ManiSkillOutputs(
                    output_action_dim=self.output_action_dim
                )
            ],
        )

        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(7, -1)
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
