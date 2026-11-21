# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""OpenPI transforms for the Tianji dual-arm contract.

Stage1 uses 20D absolute TCP poses and component-wise pose differences during
training. Quantile normalization precedes the model padding to width 32.
"""

from __future__ import annotations

import dataclasses
import pathlib

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.envs.realworld.tianji.tianji_contract import project_pose


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class TianjiInputs(transforms.DataTransformFn):
    """Map standardized Tianji observations to Pi0/Pi05 inputs."""

    model_type: _model.ModelType
    model_action_dim: int

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != 20:
            raise ValueError(f"Tianji expects a 20D state, got {tuple(state.shape)}")

        base_image = _parse_image(data["observation/image"])
        extra = _parse_image(data["observation/extra_view_image"])
        if extra.ndim != 4 or extra.shape[0] != 2:
            raise ValueError(
                "Tianji expects two wrist images with shape [2,H,W,C] or "
                f"[2,C,H,W], got {tuple(extra.shape)}"
            )
        wrist_left = _parse_image(extra[0])
        wrist_right = _parse_image(extra[1])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_left,
                "right_wrist_0_rgb": wrist_right,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != 20:
                raise ValueError(
                    f"Tianji expects 20D actions, got {tuple(actions.shape)}"
                )
            actions = actions.copy()
            for start in (0, 10):
                actions[..., start : start + 9] -= state[..., None, start : start + 9]
            inputs["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )
        return inputs


@dataclasses.dataclass(frozen=True)
class TianjiOutputs(transforms.DataTransformFn):
    """Undo the training-time component differences after unnormalization."""

    action_dim: int = 20

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < self.action_dim:
            raise ValueError(
                f"OpenPI output has {actions.shape[-1]} dims; Tianji needs "
                f"at least {self.action_dim}"
            )
        actions = actions[..., : self.action_dim].copy()
        state = np.asarray(data["state"])[..., :20]
        for start in (0, 10):
            actions[..., start : start + 9] += state[..., None, start : start + 9]
        return {"actions": project_pose(actions)}


@dataclasses.dataclass(frozen=True)
class TianjiDataConfig(DataConfigFactory):
    """DataConfigFactory entry for the Tianji Stage1 Pi05/RLT checkpoint."""

    default_prompt: str | None = None
    repo_id: str = "tianji_stage1"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/extra_view_image": "extra_view_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[
                TianjiInputs(
                    model_type=model_config.model_type,
                    model_action_dim=int(model_config.action_dim),
                )
            ],
            outputs=[TianjiOutputs(action_dim=20)],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
            use_quantile_norm=True,
        )


# ---- Joint fork: 16D joint Stage1 (state = image-aligned measured joints) ----
# [A J1..J7, A commanded closure, B J1..J7, B commanded closure]. Training
# (pi05_tianji_joint_30hz): joints learn the difference to the current measured
# joints, closures stay absolute; quantile normalization, padding to 32.
JOINT_DIMS = [d for a0 in (0, 8) for d in range(a0, a0 + 7)]
JOINT_GRIPPER_DIMS = (7, 15)


@dataclasses.dataclass(frozen=True)
class TianjiJointInputs(transforms.DataTransformFn):
    """Map 16D joint Tianji observations to Pi0/Pi05 inputs."""

    model_type: _model.ModelType
    model_action_dim: int

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != 16:
            raise ValueError(f"Tianji joint expects a 16D state, got {tuple(state.shape)}")
        base_image = _parse_image(data["observation/image"])
        extra = _parse_image(data["observation/extra_view_image"])
        if extra.ndim != 4 or extra.shape[0] != 2:
            raise ValueError(
                "Tianji expects two wrist images with shape [2,H,W,C] or "
                f"[2,C,H,W], got {tuple(extra.shape)}"
            )
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": _parse_image(extra[0]),
                "right_wrist_0_rgb": _parse_image(extra[1]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"])
            if actions.shape[-1] != 16:
                raise ValueError(f"Tianji joint expects 16D actions, got {tuple(actions.shape)}")
            actions = actions.copy()
            actions[..., JOINT_DIMS] -= state[..., None, JOINT_DIMS]
            inputs["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = (
                prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
            )
        return inputs


@dataclasses.dataclass(frozen=True)
class TianjiJointOutputs(transforms.DataTransformFn):
    """Absolute joints [rad] + closures in [0,1] after unnormalization."""

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.shape[-1] < 16:
            raise ValueError(f"OpenPI output has {actions.shape[-1]} dims; Tianji joint needs 16")
        actions = actions[..., :16].astype(np.float32).copy()
        state = np.asarray(data["state"])[..., :16]
        actions[..., JOINT_DIMS] += state[..., None, JOINT_DIMS]
        actions[..., JOINT_GRIPPER_DIMS] = np.clip(actions[..., JOINT_GRIPPER_DIMS], 0, 1)
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class TianjiJointDataConfig(DataConfigFactory):
    """DataConfigFactory entry for the 16D joint Tianji Stage1 checkpoint."""

    default_prompt: str | None = None
    repo_id: str = "tianji_stage1_joint"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/extra_view_image": "extra_view_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[
                TianjiJointInputs(
                    model_type=model_config.model_type,
                    model_action_dim=int(model_config.action_dim),
                )
            ],
            outputs=[TianjiJointOutputs()],
        )
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
            use_quantile_norm=True,
        )
