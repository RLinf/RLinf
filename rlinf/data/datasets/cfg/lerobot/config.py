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

"""
PyTorch implementation of OpenPI dataset configurations.

This module provides dataset configuration classes that match the original
OpenPI configuration system, designed to work with HuggingFace datasets
and PyTorch data loaders.
"""

import abc
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from .franka import FrankaEEInputs, FrankaEEOutputs
from .libero import LiberoInputs, LiberoOutputs  # noqa: F401
from .transforms import (
    AbsoluteActions,
    DataTransformFn,
    DeltaActions,
    Group,
    InjectDefaultPrompt,
    Normalize,
    PadStatesAndActions,
    RepackTransform,
    ResizeImages,
    compose,
    make_bool_mask,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset preprocessing and transforms, matching OpenPI structure."""

    repo_id: Optional[str] = None
    asset_id: Optional[str] = None
    norm_stats: Optional[dict[str, Any]] = None

    # Transforms applied in order: repack (dataset format -> common format),
    # data (robot-specific, pre-normalization), model (post-normalization).
    repack_transforms: Group = field(default_factory=Group)
    data_transforms: Group = field(default_factory=Group)
    model_transforms: Group = field(default_factory=Group)
    use_quantile_norm: bool = False

    # Keys used by the data loader to generate action sequences.
    action_sequence_keys: Sequence[str] = ("actions",)

    prompt_from_task: bool = True

    # Action dimensions to skip normalization for (e.g., gripper).
    # Example: {"actions": [9]} skips normalization for the 10th dimension.
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None

    def create_input_transform(self) -> DataTransformFn:
        """Create the complete input transform pipeline."""
        transforms = []
        transforms.extend(self.repack_transforms.inputs)
        transforms.extend(self.data_transforms.inputs)
        if self.norm_stats is not None:
            transforms.append(
                Normalize(
                    self.norm_stats,
                    self.use_quantile_norm,
                    skip_dims=self.action_norm_skip_dims,
                )
            )

        transforms.extend(self.model_transforms.inputs)

        return compose(transforms)

    def create_output_transform(self) -> DataTransformFn:
        """Create the output transform pipeline (for inference)."""
        output_transforms = []
        output_transforms.extend(self.model_transforms.outputs)
        output_transforms.extend(self.data_transforms.outputs)
        output_transforms.extend(self.repack_transforms.outputs)

        return compose(output_transforms)


@dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    """Base class for dataset configuration factories, matching OpenPI structure."""

    repo_id: str
    asset_id: Optional[str] = None
    norm_stats_dir: Optional[str] = None
    default_prompt: Optional[str] = None
    # PI05 uses quantile norm, PI0 uses z-score
    model_type: str = "pi05"
    # Set to True only for old Pi0 checkpoints trained with extra delta transform;
    # LIBERO/Franka actions are already delta in the dataset.
    extra_delta_transform: bool = False
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None

    @abc.abstractmethod
    def create(self, action_dim: int, *args, **kwargs) -> DataConfig:
        """Create a data configuration."""

    def _load_norm_stats(
        self,
        dataset_path: Union[str, Path],
        norm_stats_dir: Optional[str] = None,
        asset_id: Optional[str] = None,
        required: bool = True,
    ) -> Optional[dict[str, Any]]:
        """Load normalization statistics following OpenPI's pattern.

        Search order:
        1. {norm_stats_dir}/{asset_id}/norm_stats.json (if norm_stats_dir specified)
        2. {norm_stats_dir}/norm_stats.json (if norm_stats_dir is a directory)
        3. {norm_stats_dir} (if it is a direct file path to norm_stats.json)
        4. {dataset_path}/meta/norm_stats.json (fallback)

        Args:
            required: If True, raise FileNotFoundError when stats not found.
                      If False, return None (useful for stats computation).
        """
        if norm_stats_dir:
            norm_stats_path = Path(norm_stats_dir)
            candidates = []

            if norm_stats_path.is_file():
                candidates.append(norm_stats_path)
            else:
                effective_asset_id = asset_id or Path(dataset_path).name
                candidates.append(
                    norm_stats_path / effective_asset_id / "norm_stats.json"
                )
                candidates.append(norm_stats_path / "norm_stats.json")

            for stats_file in candidates:
                if stats_file.exists():
                    with open(stats_file, "r") as f:
                        stats = json.load(f)
                    logger.info(
                        f"Loaded normalization stats from norm_stats dir: {stats_file}"
                    )
                    return stats["norm_stats"]

        meta_dir = Path(dataset_path) / "meta"
        stats_file = meta_dir / "norm_stats.json"
        if stats_file.exists():
            with open(stats_file, "r") as f:
                stats = json.load(f)
            logger.info(f"Loaded normalization stats from dataset: {stats_file}")
            return stats["norm_stats"]

        if required:
            raise FileNotFoundError(f"No normalization stats found for {dataset_path}")
        return None


@dataclass(frozen=True)
class LiberoDataConfig(DataConfigFactory):
    """Configuration factory for Libero datasets.

    This config matches OpenPI's LeRobotLiberoDataConfig. Key settings:
    - extra_delta_transform: LIBERO actions are already delta in the dataset, so this
      should be False for pi05_libero. Only set to True for old Pi0 checkpoints that
      were trained with an extra delta transform.
    - use_quantile_norm: Should be True for PI05 models, False for PI0 models.
      OpenPI determines this based on model_type != PI0.
    - norm_stats_dir: Path to config-specific norm_stats (e.g., norm_stats/pi0_libero/)

    """

    def create(
        self, action_dim: int, skip_norm_stats: bool = False, *args, **kwargs
    ) -> DataConfig:
        """Create Libero dataset configuration.

        Args:
            action_dim: Action dimension for padding.
            skip_norm_stats: If True, skip loading norm stats (useful for stats computation).
        """

        repack_keys = {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
            "episode_index": "episode_index",
            "frame_index": "frame_index",
        }
        repack_transforms = Group(inputs=[RepackTransform(repack_keys)])

        data_transforms = Group(
            inputs=[LiberoInputs(mask_padding=True, model_type=self.model_type)],
            outputs=[LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[DeltaActions(delta_action_mask)],
                outputs=[AbsoluteActions(delta_action_mask)],
            )

        model_transforms_list = [
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
            PadStatesAndActions(model_action_dim=action_dim),
        ]

        model_transforms = Group(inputs=model_transforms_list)

        norm_stats = self._load_norm_stats(
            self.repo_id,
            self.norm_stats_dir,
            self.asset_id,
            required=not skip_norm_stats,
        )

        use_quantile_norm = self.model_type.lower() not in ["pi0"]

        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=use_quantile_norm,
            action_sequence_keys=["actions"],
            prompt_from_task=True,
            action_norm_skip_dims=self.action_norm_skip_dims,
        )


@dataclass(frozen=True)
class LiberoV2DataConfig(DataConfigFactory):
    """Configuration factory for Libero datasets in LeRobot v2.1 format.

    Key differences from v2.0:
    - observation.images.image instead of image
    - observation.images.wrist_image instead of wrist_image
    - observation.state instead of state
    - action (singular) instead of actions (plural)
    """

    def create(
        self, action_dim: int, skip_norm_stats: bool = False, *args, **kwargs
    ) -> DataConfig:
        """Create Libero v2.1 dataset configuration."""

        repack_keys = {
            "observation/image": "observation.images.image",
            "observation/wrist_image": "observation.images.wrist_image",
            "observation/state": "observation.state",
            "actions": "action",  # v2.1 uses singular 'action'
            "prompt": "prompt",
            "episode_index": "episode_index",
            "frame_index": "frame_index",
        }
        repack_transforms = Group(inputs=[RepackTransform(repack_keys)])

        data_transforms = Group(
            inputs=[LiberoInputs(mask_padding=True, model_type=self.model_type)],
            outputs=[LiberoOutputs()],
        )

        if self.extra_delta_transform:
            delta_action_mask = make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[DeltaActions(delta_action_mask)],
                outputs=[AbsoluteActions(delta_action_mask)],
            )

        model_transforms_list = [
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
            PadStatesAndActions(model_action_dim=action_dim),
        ]

        model_transforms = Group(inputs=model_transforms_list)

        norm_stats = self._load_norm_stats(
            self.repo_id,
            self.norm_stats_dir,
            self.asset_id,
            required=not skip_norm_stats,
        )
        use_quantile_norm = self.model_type.lower() not in ["pi0"]

        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=use_quantile_norm,
            action_sequence_keys=["actions"],
            prompt_from_task=True,
            action_norm_skip_dims=self.action_norm_skip_dims,
        )


@dataclass(frozen=True)
class FrankaDataConfig(DataConfigFactory):
    """Configuration factory for Franka datasets.

    Key differences from Libero:
    - Only 1 camera (base_0_rgb); left_wrist and right_wrist are zero-padded
    - State dim is dataset-dependent; padding is handled by PadStatesAndActions
    - action_train_with_rotation_6d controls action dim (10 vs 7) and delta mask

    Bug fix: action_train_with_rotation_6d now actually affects output slicing
    and delta mask. The original OpenPI code declared the flag but always sliced to 7.
    """

    action_train_with_rotation_6d: bool = False

    def create(
        self, action_dim: int, skip_norm_stats: bool = False, *args, **kwargs
    ) -> DataConfig:
        repack_keys = {
            "observation/image": "image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
            "episode_index": "episode_index",
            "frame_index": "frame_index",
        }
        repack_transforms = Group(inputs=[RepackTransform(repack_keys)])

        data_transforms = Group(
            inputs=[
                FrankaEEInputs(
                    mask_padding=True,
                    model_type=self.model_type,
                    action_train_with_rotation_6d=self.action_train_with_rotation_6d,
                )
            ],
            outputs=[
                FrankaEEOutputs(
                    action_train_with_rotation_6d=self.action_train_with_rotation_6d,
                )
            ],
        )

        # rotation_6d=True: 10-dim [x,y,z,rot6d(6),gripper] -> mask(9, -1)
        # rotation_6d=False: 7-dim [x,y,z,rx,ry,rz,gripper] -> mask(6, -1)
        if self.extra_delta_transform:
            if self.action_train_with_rotation_6d:
                delta_action_mask = make_bool_mask(9, -1)
            else:
                delta_action_mask = make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[DeltaActions(delta_action_mask)],
                outputs=[AbsoluteActions(delta_action_mask)],
            )

        model_transforms = Group(
            inputs=[
                InjectDefaultPrompt(self.default_prompt),
                ResizeImages(224, 224),
                PadStatesAndActions(model_action_dim=action_dim),
            ]
        )

        norm_stats = self._load_norm_stats(
            self.repo_id,
            self.norm_stats_dir,
            self.asset_id,
            required=not skip_norm_stats,
        )
        use_quantile_norm = self.model_type.lower() not in ["pi0"]

        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=use_quantile_norm,
            action_sequence_keys=["actions"],
            prompt_from_task=True,
            action_norm_skip_dims=self.action_norm_skip_dims,
        )


DATASET_CONFIGS = {
    "libero": LiberoDataConfig,
    "libero_v2": LiberoV2DataConfig,  # LeRobot v2.1 format (no_noops, _lerobot suffix)
    "franka": FrankaDataConfig,
}


def get_dataset_config(dataset_type: str, repo_id: str, **kwargs) -> DataConfigFactory:
    """Get a dataset configuration factory by type."""
    if dataset_type not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. Available: {list(DATASET_CONFIGS.keys())}"
        )

    return DATASET_CONFIGS[dataset_type](repo_id, **kwargs)


def detect_robot_type(dataset_path: str) -> str:
    """Auto-detect robot type from dataset path/name."""
    path_lower = dataset_path.lower()

    if "franka" in path_lower:
        return "franka"
    if "libero" in path_lower:
        if "no_noops" in path_lower or path_lower.endswith("_lerobot"):
            return "libero_v2"
        return "libero"
    raise ValueError(f"Unknown robot type for dataset: {dataset_path}")


def create_data_config_factory(
    dataset_path: str,
    robot_type: Optional[str] = None,
    model_type: Optional[str] = None,
    default_prompt: Optional[str] = None,
    extra_delta_transform: bool = False,
    norm_stats_dir: Optional[str] = None,
    asset_id: Optional[str] = None,
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None,
    **kwargs,
) -> DataConfigFactory:
    """Create a DataConfigFactory from configuration parameters."""
    if robot_type is None:
        robot_type = detect_robot_type(dataset_path)

    robot_type = robot_type.lower()
    logger.info(f"Creating data config factory for robot type: {robot_type}")

    common_params = {
        "default_prompt": default_prompt,
        "model_type": model_type or "pi05",
        "extra_delta_transform": extra_delta_transform,
        "action_norm_skip_dims": action_norm_skip_dims,
    }

    if robot_type == "libero":
        return LiberoDataConfig(
            repo_id=dataset_path,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
            **common_params,
        )
    elif robot_type == "libero_v2":
        return LiberoV2DataConfig(
            repo_id=dataset_path,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
            **common_params,
        )
    elif robot_type == "franka":
        return FrankaDataConfig(
            repo_id=dataset_path,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
            action_train_with_rotation_6d=kwargs.get(
                "action_train_with_rotation_6d", False
            ),
            **common_params,
        )
    else:
        raise ValueError(f"Invalid robot type: {robot_type}")


def create_data_config_factory_from_dict(config: dict[str, Any]) -> DataConfigFactory:
    """Create a DataConfigFactory from a configuration dictionary."""
    dataset_path = (
        config.get("dataset_name")
        or config.get("dataset_path")
        or config.get("repo_id")
    )
    if not dataset_path:
        raise ValueError(
            "Config must contain 'dataset_name', 'dataset_path', or 'repo_id'"
        )

    return create_data_config_factory(
        dataset_path=dataset_path,
        robot_type=config.get("robot_type"),
        model_type=config.get("model_type"),
        default_prompt=config.get("default_prompt"),
        extra_delta_transform=config.get("extra_delta_transform", False),
        norm_stats_dir=config.get("norm_stats_dir"),
        asset_id=config.get("asset_id"),
        action_norm_skip_dims=config.get("action_norm_skip_dims"),
    )
