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

from .io_processing.aloha import AlohaInputs, AlohaOutputs
from .io_processing.droid import DroidInputs, DroidOutputs
from .io_processing.franka import FrankaInputs, FrankaOutputs
from .io_processing.franka_3cam import FrankaInputs as Franka3CamInputs
from .io_processing.franka_3cam import FrankaOutputs as Franka3CamOutputs
from .io_processing.libero import LiberoInputs, LiberoOutputs
from .transforms import (
    AbsoluteActions,
    DataTransformFn,
    DeltaActions,
    FASTTokenizerTransform,
    Group,
    InjectDefaultPrompt,
    LiberoEcotBBoxReasoningTransform,
    LiberoEcotPointingReasoningTransform,
    LiberoEcotSubtaskReasoningTransform,
    Normalize,
    PadStatesAndActions,
    PromptFromLeRobotTask,
    RepackTransform,
    ResizeImages,
    SubtaskReasoningTransform,
    compose,
    load_task_descriptions,
    make_bool_mask,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    """Configuration for dataset preprocessing and transforms, matching OpenPI structure."""

    # LeRobot repo id. If None, fake data will be created.
    repo_id: Optional[str] = None
    # Directory within the assets directory containing the data assets.
    asset_id: Optional[str] = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: Optional[dict[str, Any]] = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: Group = field(default_factory=Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized.
    data_transforms: Group = field(default_factory=Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: Group = field(default_factory=Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = True

    # Action dimensions to skip normalization for (e.g., gripper). Maps key name to list of dimension indices.
    # Example: {"actions": [9]} skips normalization for the 10th dimension (gripper) of actions.
    # Empty dict or None means normalize all dimensions (default behavior for backward compatibility).
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None

    def create_input_transform(self) -> DataTransformFn:
        """Create the complete input transform pipeline."""
        transforms = []
        transforms.extend(self.repack_transforms.inputs)
        transforms.extend(self.data_transforms.inputs)
        # Add normalization if stats are available
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
        # Output transforms are applied in reverse order
        output_transforms = []
        output_transforms.extend(self.model_transforms.outputs)
        output_transforms.extend(self.data_transforms.outputs)
        output_transforms.extend(self.repack_transforms.outputs)

        return compose(output_transforms)


@dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    """Base class for dataset configuration factories, matching OpenPI structure."""

    # The LeRobot repo id.
    repo_id: str
    # Asset ID - used to locate norm_stats in norm_stats directory (defaults to repo_id if not set)
    asset_id: Optional[str] = None
    # Norm stats directory - if set, loads norm_stats from {norm_stats_dir}/{asset_id}/norm_stats.json
    # Following OpenPI pattern: norm_stats/{config_name}/{asset_id}/norm_stats.json
    norm_stats_dir: Optional[str] = None

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: Optional[str] = None

    # Model type determines normalization: PI05 uses quantile norm, PI0 uses z-score.
    model_type: str = "pi05"  # "pi0", "pi05", or "pi0_fast"

    # Some datasets (like old Pi0 checkpoints) were trained with an extra delta transform.
    # Set to False for most cases as LIBERO/Franka actions are already delta in the dataset.
    extra_delta_transform: bool = False

    # Subtask reasoning: if True, adds prefix/response from subtask annotations
    subtask_reasoning: bool = False
    subtask_prefix: str = "Subtask:"
    # Root directory for ECOT subtask data (contains file_name/demo_id/subtasks.json)
    # Required when subtask_reasoning=True for Libero datasets
    subtask_data_root: Optional[str] = None

    # BBox reasoning: if True, adds prefix/response from bbox annotations
    bbox_reasoning: bool = False
    bbox_prefix: str = "BBox:"

    # Pointing reasoning: if True, adds prefix/response from pointing annotations
    pointing_reasoning: bool = False
    pointing_prefix: str = "Point:"

    # Visual CoT data normalization: when enabled, bbox/pointing coordinates from
    # ECOT JSON (assumed 256x256 grid) are rescaled to normalization_size
    visual_cot_data_normalization: bool = False
    normalization_size: Optional[tuple[int, int]] = None

    # FAST tokenization: if True, adds FAST action tokens alongside continuous actions
    fast_tokenize: bool = False
    fast_tokenizer_path: str = "physical-intelligence/fast"

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
        2. {dataset_path}/meta/norm_stats.json (fallback)

        This matches OpenPI where each config has its own norm_stats directory.

        Args:
            required: If True, raise FileNotFoundError when stats not found.
                      If False, return None (useful for stats computation).
        """
        # Try norm_stats directory first (OpenPI pattern)
        if norm_stats_dir:
            effective_asset_id = asset_id or Path(dataset_path).name
            stats_file = Path(norm_stats_dir) / effective_asset_id / "norm_stats.json"
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    stats = json.load(f)
                logger.info(
                    f"Loaded normalization stats from norm_stats dir: {stats_file}"
                )
                return stats["norm_stats"]

        # Fall back to dataset meta directory
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

    FAST Tokenization:
    - When fast_tokenize=True, adds FASTTokenizerTransform to model_transforms
    - This tokenizes state/actions and adds tokens alongside continuous actions
    """

    def create(
        self, action_dim: int, skip_norm_stats: bool = False, *args, **kwargs
    ) -> DataConfig:
        """Create Libero dataset configuration.

        Args:
            action_dim: Action dimension for padding.
            skip_norm_stats: If True, skip loading norm stats (useful for stats computation).
        """

        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment.
        # Matching OpenPI's LeRobotLiberoDataConfig repack_transform
        repack_keys = {
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
            # Always include episode/frame indices for ECOT-based reasoning transforms
            "episode_index": "episode_index",
            "frame_index": "frame_index",
        }
        repack_transforms = Group(inputs=[RepackTransform(repack_keys)])

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # Pass model_type to LiberoInputs for correct image mask handling (PI0_FAST vs PI0/PI05)
        data_transforms = Group(
            inputs=[LiberoInputs(mask_padding=True, model_type=self.model_type)],
            outputs=[LiberoOutputs()],
        )

        # Apply delta actions transform ONLY if extra_delta_transform is True.
        # For pi05_libero, this should be False because LIBERO actions are already delta.
        # This matches OpenPI's pi05_libero config: extra_delta_transform=False
        if self.extra_delta_transform:
            delta_action_mask = make_bool_mask(
                6, -1
            )  # First 6 dims delta, last 1 absolute
            data_transforms = data_transforms.push(
                inputs=[DeltaActions(delta_action_mask)],
                outputs=[AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms_list = [
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
            PadStatesAndActions(model_action_dim=action_dim),
        ]

        # Add subtask reasoning transform if enabled
        if self.subtask_reasoning and Path(self.repo_id).exists():
            subtask_transform = None
            if self.subtask_data_root:
                subtask_transform = (
                    LiberoEcotSubtaskReasoningTransform.from_dataset_and_ecot(
                        dataset_path=self.repo_id,
                        ecot_data_root=self.subtask_data_root,
                        prefix=self.subtask_prefix,
                    )
                )
            else:
                subtask_transform = SubtaskReasoningTransform.from_dataset_path(
                    self.repo_id, prefix=self.subtask_prefix
                )
            if subtask_transform is not None:
                model_transforms_list.append(subtask_transform)
                logger.info(
                    f"Added subtask reasoning transform with prefix '{self.subtask_prefix}'"
                )

        # Add bbox reasoning transform if enabled (uses same ECOT root as subtask)
        if (
            self.bbox_reasoning
            and self.subtask_data_root
            and Path(self.repo_id).exists()
        ):
            bbox_transform = LiberoEcotBBoxReasoningTransform.from_dataset_and_ecot(
                dataset_path=self.repo_id,
                ecot_data_root=self.subtask_data_root,
                bbox_data_root=self.subtask_data_root,
                prefix=self.bbox_prefix,
                normalize_coords=self.visual_cot_data_normalization,
                normalization_size=self.normalization_size,
            )
            if bbox_transform is not None:
                model_transforms_list.append(bbox_transform)
                logger.info(
                    f"Added bbox reasoning transform with prefix '{self.bbox_prefix}'"
                )

        # Add pointing reasoning transform if enabled (uses same ECOT root as subtask)
        if (
            self.pointing_reasoning
            and self.subtask_data_root
            and Path(self.repo_id).exists()
        ):
            pointing_transform = (
                LiberoEcotPointingReasoningTransform.from_dataset_and_ecot(
                    dataset_path=self.repo_id,
                    ecot_data_root=self.subtask_data_root,
                    points_data_root=self.subtask_data_root,
                    prefix=self.pointing_prefix,
                    normalize_coords=self.visual_cot_data_normalization,
                    normalization_size=self.normalization_size,
                )
            )
            if pointing_transform is not None:
                model_transforms_list.append(pointing_transform)
                logger.info(
                    f"Added pointing reasoning transform with prefix '{self.pointing_prefix}'"
                )

        # Add FAST tokenization to compute action tokens alongside continuous actions
        if self.fast_tokenize:
            model_transforms_list.append(
                FASTTokenizerTransform(fast_tokenizer_path=self.fast_tokenizer_path)
            )
            logger.info("Added FASTTokenizerTransform")

        model_transforms = Group(inputs=model_transforms_list)

        # Load normalization stats from norm_stats_dir (OpenPI pattern) or dataset meta
        norm_stats = self._load_norm_stats(
            self.repo_id,
            self.norm_stats_dir,
            self.asset_id,
            required=not skip_norm_stats,
        )

        # Determine use_quantile_norm based on model type (matching OpenPI logic)
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
        )


@dataclass(frozen=True)
class LiberoV2DataConfig(DataConfigFactory):
    """Configuration factory for Libero datasets in LeRobot v2.1 format.

    This config is for newer LeRobot v2.1 format datasets like:
    - libero_goal_no_noops_1.0.0_lerobot
    - libero_object_no_noops_1.0.0_lerobot
    - libero_spatial_no_noops_1.0.0_lerobot
    - libero_10_no_noops_1.0.0_lerobot
    - libero_90_no_noops_lerobot

    Key differences from v2.0:
    - observation.images.image instead of image
    - observation.images.wrist_image instead of wrist_image
    - observation.state instead of state
    - action (singular) instead of actions (plural)

    When fast_tokenize=True, adds FASTTokenizerTransform for action tokenization.
    """

    def create(
        self, action_dim: int, skip_norm_stats: bool = False, *args, **kwargs
    ) -> DataConfig:
        """Create Libero v2.1 dataset configuration.

        Args:
            action_dim: Action dimension for padding.
            skip_norm_stats: If True, skip loading norm stats (useful for stats computation).
        """

        # LeRobot v2.1 format key mapping
        repack_keys = {
            "observation/image": "observation.images.image",
            "observation/wrist_image": "observation.images.wrist_image",
            "observation/state": "observation.state",
            "actions": "action",  # v2.1 uses singular 'action'
            "prompt": "prompt",
            # Always include episode/frame indices for ECOT-based reasoning transforms
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

        # Add subtask reasoning transform if enabled
        if self.subtask_reasoning and Path(self.repo_id).exists():
            subtask_transform = None
            if self.subtask_data_root:
                subtask_transform = (
                    LiberoEcotSubtaskReasoningTransform.from_dataset_and_ecot(
                        dataset_path=self.repo_id,
                        ecot_data_root=self.subtask_data_root,
                        prefix=self.subtask_prefix,
                    )
                )
            else:
                subtask_transform = SubtaskReasoningTransform.from_dataset_path(
                    self.repo_id, prefix=self.subtask_prefix
                )
            if subtask_transform is not None:
                model_transforms_list.append(subtask_transform)
                logger.info(
                    f"Added subtask reasoning transform with prefix '{self.subtask_prefix}'"
                )

        # Add bbox reasoning transform if enabled (uses same ECOT root as subtask)
        if (
            self.bbox_reasoning
            and self.subtask_data_root
            and Path(self.repo_id).exists()
        ):
            bbox_transform = LiberoEcotBBoxReasoningTransform.from_dataset_and_ecot(
                dataset_path=self.repo_id,
                ecot_data_root=self.subtask_data_root,
                bbox_data_root=self.subtask_data_root,
                prefix=self.bbox_prefix,
                normalize_coords=self.visual_cot_data_normalization,
                normalization_size=self.normalization_size,
            )
            if bbox_transform is not None:
                model_transforms_list.append(bbox_transform)
                logger.info(
                    f"Added bbox reasoning transform with prefix '{self.bbox_prefix}'"
                )

        # Add pointing reasoning transform if enabled (uses same ECOT root as subtask)
        if (
            self.pointing_reasoning
            and self.subtask_data_root
            and Path(self.repo_id).exists()
        ):
            pointing_transform = (
                LiberoEcotPointingReasoningTransform.from_dataset_and_ecot(
                    dataset_path=self.repo_id,
                    ecot_data_root=self.subtask_data_root,
                    points_data_root=self.subtask_data_root,
                    prefix=self.pointing_prefix,
                    normalize_coords=self.visual_cot_data_normalization,
                    normalization_size=self.normalization_size,
                )
            )
            if pointing_transform is not None:
                model_transforms_list.append(pointing_transform)
                logger.info(
                    f"Added pointing reasoning transform with prefix '{self.pointing_prefix}'"
                )

        # Add FAST tokenization to compute action tokens alongside continuous actions
        if self.fast_tokenize:
            model_transforms_list.append(
                FASTTokenizerTransform(fast_tokenizer_path=self.fast_tokenizer_path)
            )
            logger.info("Added FASTTokenizerTransform")

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
        )


@dataclass(frozen=True)
class DroidDataConfig(DataConfigFactory):
    """Configuration factory for DROID datasets."""

    def create(self, action_dim: int, *args, **kwargs) -> DataConfig:
        """Create DROID dataset configuration."""

        # Load task descriptions if using prompt from task
        tasks = {}
        if Path(self.repo_id).exists():
            tasks = load_task_descriptions(self.repo_id)

        # Repack transforms - map DROID dataset keys to standard format
        repack_transforms = Group(
            inputs=[
                RepackTransform(
                    {
                        "observation/image": "observation/exterior_image_1_left",
                        "observation/wrist_image": "observation/wrist_image_left",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "action",  # Map singular 'action' to plural 'actions'
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data transforms - DROID-specific processing
        data_transforms = Group(
            inputs=[DroidInputs()],
            outputs=[DroidOutputs()],
        )

        # DROID typically uses delta actions for all dimensions
        delta_action_mask = make_bool_mask(7, -1)  # First 7 dims delta, last 1 absolute
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask)],
            outputs=[AbsoluteActions(delta_action_mask)],
        )

        # Model transforms - standard preprocessing
        model_transforms_list = [
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
            PadStatesAndActions(model_action_dim=action_dim),
        ]

        # Add prompt from task if tasks are available
        if tasks:
            model_transforms_list.insert(0, PromptFromLeRobotTask(tasks))

        # Add FAST tokenization if enabled
        if self.fast_tokenize:
            model_transforms_list.append(
                FASTTokenizerTransform(fast_tokenizer_path=self.fast_tokenizer_path)
            )
            logger.info("Added FASTTokenizerTransform")

        model_transforms = Group(inputs=model_transforms_list)

        # Load normalization stats
        norm_stats = (
            self._load_norm_stats(self.repo_id) if Path(self.repo_id).exists() else None
        )

        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=False,
            action_sequence_keys=["actions"],
            prompt_from_task=bool(tasks),
        )


@dataclass(frozen=True)
class AlohaDataConfig(DataConfigFactory):
    """Configuration factory for ALOHA datasets."""

    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi: bool = True

    def create(self, action_dim: int, *args, **kwargs) -> DataConfig:
        """Create ALOHA dataset configuration."""

        # Load task descriptions if using prompt from task
        tasks = {}
        if Path(self.repo_id).exists():
            tasks = load_task_descriptions(self.repo_id)

        # Repack transforms - map ALOHA dataset keys to standard format
        repack_transforms = Group(
            inputs=[
                RepackTransform(
                    {
                        "state": "state",  # ALOHA state is already in standard format
                        "images": "images",  # ALOHA images dict is already in standard format
                        "actions": "action",  # Map singular 'action' to plural 'actions'
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Data transforms - ALOHA-specific processing
        data_transforms = Group(
            inputs=[AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )

        # ALOHA typically uses delta actions for joint angles but absolute for grippers
        # First 6 dims (left arm joints) + next 6 dims (right arm joints) = 12 delta
        # Last 2 dims (grippers) = absolute
        delta_action_mask = make_bool_mask(
            12, -2
        )  # First 12 dims delta, last 2 absolute
        data_transforms = data_transforms.push(
            inputs=[DeltaActions(delta_action_mask)],
            outputs=[AbsoluteActions(delta_action_mask)],
        )

        # Model transforms - standard preprocessing
        model_transforms_list = [
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
            PadStatesAndActions(model_action_dim=action_dim),
        ]

        # Add prompt from task if tasks are available
        if tasks:
            model_transforms_list.insert(0, PromptFromLeRobotTask(tasks))

        # Add FAST tokenization if enabled
        if self.fast_tokenize:
            model_transforms_list.append(
                FASTTokenizerTransform(fast_tokenizer_path=self.fast_tokenizer_path)
            )
            logger.info("Added FASTTokenizerTransform")

        model_transforms = Group(inputs=model_transforms_list)

        # Load normalization stats
        norm_stats = (
            self._load_norm_stats(self.repo_id) if Path(self.repo_id).exists() else None
        )

        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=False,
            action_sequence_keys=["actions"],
            prompt_from_task=bool(tasks),
        )


@dataclass(frozen=True)
class FrankaDataConfig(DataConfigFactory):
    """Configuration factory for Franka robot datasets.

    This config is designed for Franka data collected with the realRL framework
    and converted to LeRobot format.

    Uses 6D rotation representation for both state and action to avoid singularities.
    Matches OpenPI's LeRobotFrankaDataConfig.
    """

    # Action sequence keys in the dataset
    action_sequence_keys: tuple = ("action",)

    # Action dimensions to skip normalization for. Maps key name to list of dimension indices.
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None

    def create(
        self, action_dim: int, skip_norm_stats: bool = False, *args, **kwargs
    ) -> DataConfig:
        """Create Franka dataset configuration."""

        # Repack transform: maps LeRobot dataset keys to the format expected by FrankaInputs
        repack_keys = {
            "observation/state/tcp_pose": "observation.state.tcp_pose",
            "observation/state/gripper_pose": "observation.state.gripper_pose",
            "observation/images/front_cam": "observation.images.front_cam",
            "observation/images/wrist_cam": "observation.images.wrist_cam",
            "actions": "action",
            "prompt": "prompt",
            "task_index": "task_index",
            "return": "return",
        }
        # Add task_idx and subtask_idx for subtask reasoning
        if self.subtask_reasoning:
            repack_keys["task_idx"] = "task_idx"
            repack_keys["subtask_idx"] = "subtask_idx"
            repack_keys["episode_index"] = "episode_index"
            repack_keys["frame_index"] = "frame_index"

        repack_transforms = Group(inputs=[RepackTransform(repack_keys)])

        # Data transforms: Applied to both training data and inference inputs
        data_transforms = Group(
            inputs=[FrankaInputs(model_type=self.model_type)],
            outputs=[FrankaOutputs()],
        )

        # Model transforms: Tokenization, resizing, padding
        model_transforms_list = [
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
            PadStatesAndActions(model_action_dim=action_dim),
        ]

        # Add subtask reasoning transform if enabled
        if self.subtask_reasoning and Path(self.repo_id).exists():
            subtask_transform = None
            if self.subtask_data_root:
                subtask_transform = (
                    LiberoEcotSubtaskReasoningTransform.from_dataset_and_ecot(
                        dataset_path=self.repo_id,
                        ecot_data_root=self.subtask_data_root,
                        prefix=self.subtask_prefix,
                    )
                )
            else:
                subtask_transform = SubtaskReasoningTransform.from_dataset_path(
                    self.repo_id, prefix=self.subtask_prefix
                )
            if subtask_transform is not None:
                model_transforms_list.append(subtask_transform)
                logger.info(
                    f"Added subtask reasoning transform with prefix '{self.subtask_prefix}'"
                )

        # Add FAST tokenization to compute action tokens alongside continuous actions
        if self.fast_tokenize:
            model_transforms_list.append(
                FASTTokenizerTransform(fast_tokenizer_path=self.fast_tokenizer_path)
            )
            logger.info("Added FASTTokenizerTransform")

        model_transforms = Group(inputs=model_transforms_list)

        # Load normalization stats from norm_stats_dir (OpenPI pattern) or dataset meta
        norm_stats = self._load_norm_stats(
            self.repo_id,
            self.norm_stats_dir,
            self.asset_id,
            required=not skip_norm_stats,
        )

        # Determine use_quantile_norm based on model type (matching OpenPI logic)
        use_quantile_norm = self.model_type.lower() not in ["pi0"]

        return DataConfig(
            repo_id=self.repo_id,
            asset_id=self.asset_id,
            norm_stats=norm_stats,
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=use_quantile_norm,
            action_sequence_keys=list(self.action_sequence_keys),
            prompt_from_task=True,
            action_norm_skip_dims=self.action_norm_skip_dims,
        )


@dataclass(frozen=True)
class Franka3CamDataConfig(DataConfigFactory):
    """Configuration factory for Franka robot datasets with 3 cameras.

    Identical to FrankaDataConfig except uses 3 cameras:
    left_cam, right_cam, wrist_cam (mapped to base_0_rgb, base_1_rgb, wrist_0_rgb).
    """

    action_sequence_keys: tuple = ("action",)
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None

    def create(
        self, action_dim: int, skip_norm_stats: bool = False, *args, **kwargs
    ) -> DataConfig:
        """Create Franka 3-camera dataset configuration."""

        repack_keys = {
            "observation/state/tcp_pose": "observation.state.tcp_pose",
            "observation/state/gripper_pose": "observation.state.gripper_pose",
            "observation/images/left_cam": "observation.images.left_cam",
            "observation/images/right_cam": "observation.images.right_cam",
            "observation/images/wrist_cam": "observation.images.wrist_cam",
            "actions": "action",
            "prompt": "prompt",
            "task_index": "task_index",
            "return": "return",
        }
        if self.subtask_reasoning:
            repack_keys["task_idx"] = "task_idx"
            repack_keys["subtask_idx"] = "subtask_idx"
            repack_keys["episode_index"] = "episode_index"
            repack_keys["frame_index"] = "frame_index"

        repack_transforms = Group(inputs=[RepackTransform(repack_keys)])

        data_transforms = Group(
            inputs=[Franka3CamInputs()],
            outputs=[Franka3CamOutputs()],
        )

        model_transforms_list = [
            InjectDefaultPrompt(self.default_prompt),
            ResizeImages(224, 224),
            PadStatesAndActions(model_action_dim=action_dim),
        ]

        if self.subtask_reasoning and Path(self.repo_id).exists():
            subtask_transform = None
            if self.subtask_data_root:
                subtask_transform = (
                    LiberoEcotSubtaskReasoningTransform.from_dataset_and_ecot(
                        dataset_path=self.repo_id,
                        ecot_data_root=self.subtask_data_root,
                        prefix=self.subtask_prefix,
                    )
                )
            else:
                subtask_transform = SubtaskReasoningTransform.from_dataset_path(
                    self.repo_id, prefix=self.subtask_prefix
                )
            if subtask_transform is not None:
                model_transforms_list.append(subtask_transform)
                logger.info(
                    f"Added subtask reasoning transform with prefix '{self.subtask_prefix}'"
                )

        if self.fast_tokenize:
            model_transforms_list.append(
                FASTTokenizerTransform(fast_tokenizer_path=self.fast_tokenizer_path)
            )
            logger.info("Added FASTTokenizerTransform")

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
            action_sequence_keys=list(self.action_sequence_keys),
            prompt_from_task=True,
            action_norm_skip_dims=self.action_norm_skip_dims,
        )


# Predefined configurations for common datasets
DATASET_CONFIGS = {
    "libero": LiberoDataConfig,
    "libero_v2": LiberoV2DataConfig,  # LeRobot v2.1 format (no_noops, _lerobot suffix)
    "droid": DroidDataConfig,
    "aloha": AlohaDataConfig,
    "franka": FrankaDataConfig,  # Franka robot with rot6d representation
    "franka_3cam": Franka3CamDataConfig,  # Franka robot with 3 cameras
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

    # Check in priority order (more specific patterns first)
    # LeRobot v2.1 format detection (no_noops or _lerobot suffix)
    if "libero" in path_lower:
        if "no_noops" in path_lower or path_lower.endswith("_lerobot"):
            return "libero_v2"
        return "libero"
    if "droid" in path_lower:
        return "droid"
    if "aloha" in path_lower:
        return "aloha"
    if "bridge" in path_lower:
        return "bridge"
    if "rt1" in path_lower or "fractal" in path_lower:
        return "rt1"
    if "franka" in path_lower:
        return "franka"
    if "simpler" in path_lower:
        if "bridge" in path_lower:
            return "bridge"
        return "rt1"

    raise ValueError(f"Unknown robot type for dataset: {dataset_path}")


def create_data_config_factory(
    dataset_path: str,
    robot_type: Optional[str] = None,
    model_type: Optional[str] = None,
    default_prompt: Optional[str] = None,
    extra_delta_transform: bool = False,
    norm_stats_dir: Optional[str] = None,
    asset_id: Optional[str] = None,
    adapt_to_pi: bool = True,
    subtask_reasoning: bool = False,
    subtask_prefix: str = "Subtask:",
    subtask_data_root: Optional[str] = None,
    bbox_reasoning: bool = False,
    bbox_prefix: str = "BBox:",
    pointing_reasoning: bool = False,
    pointing_prefix: str = "Point:",
    visual_cot_data_normalization: bool = False,
    normalization_size: Optional[tuple[int, int]] = None,
    fast_tokenize: bool = False,
    fast_tokenizer_path: str = "physical-intelligence/fast",
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None,
    **kwargs,
) -> DataConfigFactory:
    """Create a DataConfigFactory from configuration parameters."""
    if robot_type is None:
        robot_type = detect_robot_type(dataset_path)

    robot_type = robot_type.lower()
    logger.info(f"Creating data config factory for robot type: {robot_type}")

    # Common parameters inherited from DataConfigFactory base class
    common_params = {
        "default_prompt": default_prompt,
        "model_type": model_type or "pi05",
        "extra_delta_transform": extra_delta_transform,
        "subtask_reasoning": subtask_reasoning,
        "subtask_prefix": subtask_prefix,
        "subtask_data_root": subtask_data_root,
        "bbox_reasoning": bbox_reasoning,
        "bbox_prefix": bbox_prefix,
        "pointing_reasoning": pointing_reasoning,
        "pointing_prefix": pointing_prefix,
        "visual_cot_data_normalization": visual_cot_data_normalization,
        "normalization_size": normalization_size,
        "fast_tokenize": fast_tokenize,
        "fast_tokenizer_path": fast_tokenizer_path,
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
    elif robot_type == "droid":
        return DroidDataConfig(
            repo_id=dataset_path,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
            **common_params,
        )
    elif robot_type == "aloha":
        return AlohaDataConfig(
            repo_id=dataset_path,
            adapt_to_pi=adapt_to_pi,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
            **common_params,
        )
    elif robot_type == "franka":
        return FrankaDataConfig(
            repo_id=dataset_path,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
            action_norm_skip_dims=action_norm_skip_dims,
            **common_params,
        )
    elif robot_type == "franka_3cam":
        return Franka3CamDataConfig(
            repo_id=dataset_path,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
            action_norm_skip_dims=action_norm_skip_dims,
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
        adapt_to_pi=config.get("adapt_to_pi", True),
        subtask_reasoning=config.get("subtask_reasoning", False),
        subtask_prefix=config.get("subtask_prefix", "Subtask:"),
        subtask_data_root=config.get("subtask_data_root"),
        bbox_reasoning=config.get("bbox_reasoning", False),
        bbox_prefix=config.get("bbox_prefix", "BBox:"),
        pointing_reasoning=config.get("pointing_reasoning", False),
        pointing_prefix=config.get("pointing_prefix", "Point:"),
        visual_cot_data_normalization=config.get(
            "visual_cot_data_normalization", False
        ),
        normalization_size=config.get("normalization_size"),
        fast_tokenize=config.get("fast_tokenize", False),
        fast_tokenizer_path=config.get(
            "fast_tokenizer_path", "physical-intelligence/fast"
        ),
        action_norm_skip_dims=config.get("action_norm_skip_dims"),
    )
