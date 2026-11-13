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
from pathlib import Path

import openpi.models.model as model
import openpi.transforms as transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

from rlinf.models.embodiment.openpi.policies.bench2dex_policy import (
    Bench2DexInputs,
    Bench2DexNormalize,
    Bench2DexOutputs,
    Bench2DexUnnormalize,
)


@dataclasses.dataclass(frozen=True)
class Bench2DexDataConfig(DataConfigFactory):
    """Use checkpoint quantiles and dimensions supplied by the model factory."""

    action_dim: int | None = None

    def create(
        self, assets_dirs: Path, model_config: model.BaseModelConfig
    ) -> DataConfig:
        if (
            self.action_dim is None
            or not 0 < self.action_dim <= model_config.action_dim
        ):
            raise ValueError(
                f"Expected 0 < Bench2Dex action_dim <= {model_config.action_dim}, got {self.action_dim}"
            )
        base = self.create_base_config(assets_dirs, model_config)
        if base.norm_stats is None:
            raise ValueError("Bench2Dex requires the checkpoint's norm_stats.json")
        for key in ("state", "actions"):
            stats = base.norm_stats[key]
            if len(stats.mean) not in (self.action_dim, model_config.action_dim):
                raise ValueError(
                    f"{key} norm width {len(stats.mean)} does not match environment/model dimensions"
                )
            if stats.q01 is None or stats.q99 is None:
                raise ValueError(f"Bench2Dex {key} normalization requires q01 and q99")
        model_transforms = ModelTransformFactory()(model_config)
        # Normalize real joints with either D- or M-wide checkpoint stats, then
        # pad once before tokenizing state. Padding coordinates stay zero.
        model_transforms = dataclasses.replace(
            model_transforms,
            inputs=[
                transforms.PadStatesAndActions(model_config.action_dim),
                *[
                    transform
                    for transform in model_transforms.inputs
                    if not isinstance(transform, transforms.PadStatesAndActions)
                ],
            ],
        )
        return dataclasses.replace(
            base,
            use_quantile_norm=True,
            data_transforms=transforms.Group(
                inputs=[
                    Bench2DexInputs(self.action_dim, self.action_dim),
                    Bench2DexNormalize(base.norm_stats, use_quantiles=True),
                ],
                outputs=[
                    Bench2DexUnnormalize(base.norm_stats, use_quantiles=True),
                    Bench2DexOutputs(self.action_dim),
                ],
            ),
            model_transforms=model_transforms,
        )
