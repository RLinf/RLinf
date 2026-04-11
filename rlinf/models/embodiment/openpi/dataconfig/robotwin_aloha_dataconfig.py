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

from rlinf.models.embodiment.openpi.policies import aloha_policy


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    """Data configuration for the RoboTwin (Aloha) dataset in LeRobot v2.1 Parquet format."""

    # Default prompt to use if the dataset does not contain prompt information.
    default_prompt: str | None = None

    # If True, converts absolute joint actions to delta actions (relative to the current state).
    # This is required for the Pi0 model which operates on relative actions.
    extra_delta_transform: bool = True

    # If True, maps the data to the internal Pi0 space (e.g., flipping specific joints).
    # Set to False for standard Aloha data that does not require realignment.
    adapt_to_pi: bool = True

    # Configuration to map dataset keys to the keys expected by the model.
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=lambda: _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_high",
                            "cam_left_wrist": "observation.images.cam_left_wrist",
                            "cam_right_wrist": "observation.images.cam_right_wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
    )

    def generate_observations(
        self, image: np.ndarray, state: np.ndarray, prompt: str
    ) -> dict:
        return {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:  # Apply Aloha-specific policy transforms (e.g. normalization).
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        # Apply Delta Action transformation if enabled.
        # The mask corresponds to Aloha's 14-dim action space:
        # [Left Arm (6 joints, 1 gripper), Right Arm (6 joints, 1 gripper)].
        # We apply deltas to joints (True) but keep grippers absolute (False).
        if self.extra_delta_transform:
            delta_action_mask = np.array(
                [True] * 6 + [False] + [True] * 6 + [False],
                dtype=bool,
            )

            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Apply model-specific transforms (e.g., Tokenization, Resizing).
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        # Construct and return the final DataConfig.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
