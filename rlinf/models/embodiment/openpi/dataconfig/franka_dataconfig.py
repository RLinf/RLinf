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

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import franka_policy


@dataclasses.dataclass(frozen=True)
class CustomDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # Finally we will use delta actions to train, but we can input abs_action(get delta for training via abs_action-state) or delta_action(no other process)
    extra_delta_transform: bool = True  # False for additional process(abs_action - state) to get delta action for training
    # train actions using rotation_6d
    action_train_with_rotation_6d: bool = False

    def generate_observations(
        image: np.ndarray, state: np.ndarray, prompt: str
    ) -> dict:
        """Creates an input example for the Franka policy."""
        return {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                franka_policy.FrankaEEInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    action_train_with_rotation_6d=self.action_train_with_rotation_6d,
                )
            ],
            outputs=[
                franka_policy.FrankaEEOutputs(
                    action_train_with_rotation_6d=self.action_train_with_rotation_6d
                )
            ],
        )

        if not self.extra_delta_transform:  # for abs_action
            delta_action_mask = _transforms.make_bool_mask(
                9, -1
            )  # [True]x9 + [False]x1, [x,y,z,rotation_6d,gripper] for 10 dim
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
