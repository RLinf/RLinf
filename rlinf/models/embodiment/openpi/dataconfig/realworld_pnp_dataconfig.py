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
"""Data config for real-world Franka PnP tasks with multi-camera support.

Paired with :class:`~rlinf.models.embodiment.openpi.policies.realworld_policy.RealworldInputs`
to handle 19D→7D state selection and flexible Pi0 camera-slot mapping.
"""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import realworld_policy


@dataclasses.dataclass(frozen=True)
class LeRobotRealworldPnPDataConfig(DataConfigFactory):
    """LeRobot data config for real-world Franka pick-and-place tasks.

    Supports datasets with 19D state vectors and up to 3 camera views.
    State dimension selection and camera-to-Pi0-slot mapping are configurable.
    """

    extra_delta_transform: bool = False

    # Indices to select from the dataset state vector (19D → 7D).
    # Set to None when the dataset already stores 7D state.
    state_indices: tuple[int, ...] | None = (4, 5, 6, 7, 8, 9, 0)

    # Parquet column names of extra camera images in the LeRobot dataset.
    extra_image_keys: tuple[str, ...] = ()

    # Mapping of Pi0 image slots to observation keys after repack.
    pi0_slot_keys: tuple[str | None, str | None, str | None] = (
        "observation/extra_image_0",
        "observation/image",
        "observation/extra_image_1",
    )

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_mapping = {
            "observation/image": "image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
        for i, col in enumerate(self.extra_image_keys):
            repack_mapping[f"observation/extra_image_{i}"] = col

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )

        data_transforms = _transforms.Group(
            inputs=[
                realworld_policy.RealworldInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    state_indices=self.state_indices,
                    pi0_slot_keys=self.pi0_slot_keys,
                )
            ],
            outputs=[realworld_policy.RealworldOutputs()],
        )

        if not self.extra_delta_transform:
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
