# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""OpenPI data configuration for LeRobot SO-101 demonstrations."""

from __future__ import annotations

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import so101_policy


@dataclasses.dataclass(frozen=True)
class LeRobotSO101DataConfig(DataConfigFactory):
    """Repack SO-101 joint demonstrations into the OpenPI input contract."""

    default_prompt: str | None = None
    joint_dim: int = so101_policy.SO101_STATE_DIM

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        """Build data config for SO-101 LeRobot dataset.

        Maps flat LeRobot fields to OpenPI conventions and configures
        SO-101-specific transforms for 6-DoF state and action.
        """
        repack_transforms = _transforms.Group(
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
                so101_policy.SO101Inputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    joint_dim=self.joint_dim,
                )
            ],
            outputs=[so101_policy.SO101Outputs(joint_dim=self.joint_dim)],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
        )
