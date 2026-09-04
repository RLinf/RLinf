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

from __future__ import annotations

import dataclasses
import pathlib
from typing import Any

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.training.optimizer as _optimizer
from openpi import transforms as _transforms
from openpi.training.config import (
    AssetsConfig,
    DataConfig,
    DataConfigFactory,
    TrainConfig,
)
from typing_extensions import override

from rlinf.models.embodiment.streamingvla.transforms import (
    PadStreamingVLAStatesActions,
    StreamingVLALiberoInputs,
    StreamingVLALiberoOutputs,
)


@dataclasses.dataclass(frozen=True)
class StreamingVLAModelTransformFactory:
    """Create model transforms that preserve StreamingVLA action states."""

    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Build the tokenization, resize, and padding transform group."""
        return _transforms.Group(
            inputs=[
                _transforms.InjectDefaultPrompt(self.default_prompt),
                _transforms.ResizeImages(224, 224),
                _transforms.TokenizePrompt(
                    _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                    discrete_state_input=getattr(
                        model_config, "discrete_state_input", False
                    ),
                ),
                PadStreamingVLAStatesActions(model_config.action_dim),
            ],
        )


@dataclasses.dataclass(frozen=True)
class LeRobotStreamingVLALiberoDataConfig(DataConfigFactory):
    """LIBERO LeRobot data configuration for StreamingVLA SFT."""

    extra_delta_transform: bool = False
    action_env_dim: int = 7
    use_action_states: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        """Create the isolated LIBERO transform pipeline."""
        repack_mapping = {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
        if self.use_action_states:
            repack_mapping["observation/action_states"] = "action_states"
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )
        data_transforms = _transforms.Group(
            inputs=[StreamingVLALiberoInputs(model_type=model_config.model_type)],
            outputs=[StreamingVLALiberoOutputs(action_env_dim=self.action_env_dim)],
        )
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        base_config = self.create_base_config(assets_dirs, model_config)
        replace_kwargs = {
            "repack_transforms": repack_transform,
            "data_transforms": data_transforms,
            "model_transforms": StreamingVLAModelTransformFactory()(model_config),
        }
        return dataclasses.replace(base_config, **replace_kwargs)


_PI05_LIBERO_SFP_MODEL = pi0_config.Pi0Config(
    pi05=True,
    action_horizon=10,
    discrete_state_input=False,
)
# RLinf currently packages an older OpenPI Pi0Config without this field. The
# StreamingVLA model consumes it, so attach the canonical reference value
# explicitly instead of weakening the reference configuration.
object.__setattr__(_PI05_LIBERO_SFP_MODEL, "use_sfp", True)


_CONFIGS = [
    TrainConfig(
        # Keep the algorithmic values aligned with
        # openpi/training/config.py::pi05_libero_sfp. User paths are supplied
        # exclusively through the RLinf YAML.
        name="pi05_libero_sfp",
        model=_PI05_LIBERO_SFP_MODEL,
        data=LeRobotStreamingVLALiberoDataConfig(
            repo_id="fake",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(),
            extra_delta_transform=False,
            use_action_states=True,
        ),
        batch_size=16,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=1e-5,
            decay_steps=1_000_000,
            decay_lr=1e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        pytorch_weight_path=None,
        num_train_steps=10_000,
        log_interval=10,
    ),
]
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def _override_with_model_path(config: TrainConfig, model_path: str) -> TrainConfig:
    return dataclasses.replace(config, pytorch_weight_path=model_path)


def _override_with_data_kwargs(
    config: TrainConfig, data_kwargs: dict[str, Any]
) -> TrainConfig:
    data_kwargs = dict(data_kwargs)
    assets_kwargs = data_kwargs.pop("assets", None)
    data_config = config.data
    if assets_kwargs is not None:
        data_config = dataclasses.replace(
            data_config,
            assets=dataclasses.replace(data_config.assets, **dict(assets_kwargs)),
        )
    data_config = dataclasses.replace(data_config, **data_kwargs)
    return dataclasses.replace(config, data=data_config)


def get_streamingvla_config(
    config_name: str,
    model_path: str | None = None,
    batch_size: int | None = None,
    data_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
) -> TrainConfig:
    """Return a fresh StreamingVLA training config with optional overrides."""
    if config_name not in _CONFIGS_DICT:
        raise ValueError(f"StreamingVLA config {config_name!r} not found.")
    config = _CONFIGS_DICT[config_name]
    # Return a distinct model object so immutable overrides in one worker do not
    # contaminate the next configuration request.
    model_config = dataclasses.replace(config.model)
    object.__setattr__(
        model_config, "use_sfp", bool(getattr(config.model, "use_sfp", False))
    )
    config = dataclasses.replace(config, model=model_config)
    if model_path is not None:
        config = _override_with_model_path(config, model_path)
    if data_kwargs is not None:
        config = _override_with_data_kwargs(config, data_kwargs)
    if batch_size is not None:
        config = dataclasses.replace(config, batch_size=batch_size)
    if seed is not None:
        config = dataclasses.replace(config, seed=int(seed))
    return config
