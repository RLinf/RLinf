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

"""OpenPI data transforms for GenieSim G2 dual-arm checkpoints."""

import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import geniesim_policy


@dataclasses.dataclass(frozen=True)
class LeRobotGenieSimDualArmDataConfig(DataConfigFactory):
    """Configure the G2 dual-arm state and action transforms."""

    # Use joint deltas and absolute gripper actions, as in the SFT checkpoint.
    use_delta_joint_actions: bool = True
    output_dim: int = geniesim_policy.GENIESIM_CANONICAL_DIM
    default_prompt: str | None = None
    # Optionally mask gripper state (dimensions 14 and 15).
    mask_gripper_state: bool = False
    require_wrist_images: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        action_dim = model_config.action_dim

        # Mask the action and state dimensions beyond the two grippers.
        state_mask = np.zeros(action_dim, dtype=bool)
        action_mask = np.zeros(action_dim, dtype=bool)
        if self.mask_gripper_state:
            state_mask[14:16] = True
        state_mask[geniesim_policy.GENIESIM_CANONICAL_DIM :] = True
        action_mask[geniesim_policy.GENIESIM_CANONICAL_DIM :] = True

        data_transforms = _transforms.Group(
            inputs=[
                geniesim_policy.GenieSimInputs(
                    action_dim=action_dim,
                    state_mask=state_mask,
                    action_mask=action_mask,
                    require_wrist_images=self.require_wrist_images,
                )
            ],
            outputs=[geniesim_policy.GenieSimOutputs(output_dim=self.output_dim)],
        )

        if self.use_delta_joint_actions:
            # Keep the two grippers absolute; the masked tail follows OpenPI.
            delta_action_mask = _transforms.make_bool_mask(
                geniesim_policy.GENIESIM_NUM_ARM_JOINTS, -2, 6
            )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
