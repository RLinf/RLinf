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

from rlinf.models.embodiment.openpi.dataconfig.sfp_transforms import PadSfpActionStates
from rlinf.models.embodiment.openpi.policies import libero_policy, libero_sfp_policy


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoSfpDataConfig(DataConfigFactory):
    """LIBERO data config for Streaming Flow Policy training.

    This is :class:`~rlinf.models.embodiment.openpi.dataconfig.libero_dataconfig.LeRobotLiberoDataConfig`
    plus the ``action_states`` field SFP needs: the repack keeps it, the input
    transform passes it through, and the model transform pads it to the model
    action dimension alongside the state and actions.

    The scale-only normalization SFP also requires is not part of this config,
    because openpi builds the ``Normalize`` step itself when it assembles a
    pipeline. The SFP data loader inserts
    :class:`~rlinf.models.embodiment.openpi.dataconfig.sfp_transforms.SfpNormalize`
    in its place; see :func:`rlinf.data.datasets.openpi_rlinf.libero.build_libero_sfp_sft_dataloader`.
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
                        "observation/state": "state",
                        "observation/action_states": "action_states",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                libero_sfp_policy.LiberoSfpInputs(model_type=model_config.model_type)
            ],
            outputs=[libero_policy.LiberoOutputs()],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config).push(
            inputs=[PadSfpActionStates(model_config.action_dim)]
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
