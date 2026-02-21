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
PyTorch implementation of OpenPI transforms for dataset preprocessing.

This module provides PyTorch-based transforms that match the functionality
of the original JAX-based OpenPI transforms, designed to work with
HuggingFace datasets and the broader PyTorch ecosystem.
"""

import glob
import io
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class DataTransformFn:
    """Base class for data transforms."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply transformation to the data.

        Args:
            data: Dictionary containing the data to transform

        Returns:
            Transformed data dictionary
        """
        raise NotImplementedError


@dataclass
class Group:
    """A group of transforms matching OpenPI's structure."""

    # Transforms that are applied to the model input data.
    inputs: Sequence[DataTransformFn] = ()

    # Transforms that are applied to the model output data.
    outputs: Sequence[DataTransformFn] = ()

    def push(
        self,
        *,
        inputs: Sequence[DataTransformFn] = (),
        outputs: Sequence[DataTransformFn] = (),
    ) -> "Group":
        """Append transforms to the group and return a new group.

        Args:
            inputs: Appended to the *end* of the current input transforms.
            outputs: Appended to the *beginning* of the current output transforms.

        Returns:
            A new group with the appended transforms.
        """
        return Group(inputs=(*self.inputs, *inputs), outputs=(*outputs, *self.outputs))


class CompositeTransform(DataTransformFn):
    """Applies a sequence of transforms in order."""

    def __init__(self, transforms: list[DataTransformFn]):
        self.transforms = transforms

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(list(transforms))


class RepackTransform(DataTransformFn):
    """Repacks an input dictionary into a new dictionary structure.

    This matches the OpenPI RepackTransform functionality, allowing us to
    remap keys from dataset-specific formats to a common format.
    """

    def __init__(self, structure: dict[str, str], passthrough_unmapped: bool = False):
        """
        Args:
            structure: Mapping from new keys to old keys (flattened paths)
            passthrough_unmapped: If True, keys not in structure are passed through unchanged.
                                  Useful for RL datasets with additional keys like action_chunk, etc.
        """
        self.structure = structure
        self.passthrough_unmapped = passthrough_unmapped
        self._mapped_old_keys = set(structure.values())

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = {}

        # Simple flat key mapping for LeRobot datasets
        for new_key, old_key in self.structure.items():
            if old_key in data:
                result[new_key] = data[old_key]
            elif not self.passthrough_unmapped:
                # Only warn if passthrough_unmapped is False (strict mode)
                # When passthrough_unmapped=True, missing mapped keys are expected
                # (e.g., RL datasets restructure "action" to "action_chunk")
                logger.warning(f"Warning: Key '{old_key}' not found in data")

        # Pass through all reasoning data (observation.reasoning.*)
        for key, value in data.items():
            if key.startswith("observation.reasoning."):
                result[key] = value

        # Optionally pass through unmapped keys (for RL datasets with extra keys)
        if self.passthrough_unmapped:
            for key, value in data.items():
                if key not in self._mapped_old_keys and key not in result:
                    result[key] = value

        return result

    def _flatten_dict(
        self, d: dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


class InjectDefaultPrompt(DataTransformFn):
    """Injects a default prompt if none is present."""

    def __init__(self, prompt: Optional[str]):
        self.prompt = prompt

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = self.prompt
        return data


class ResizeImages(DataTransformFn):
    """Resizes images to the specified dimensions.

    Works with images in the "image" or "images" dict.
    Handles both:
    - torch.Tensor in CHW format (from LiberoInputs)
    - np.ndarray in HWC format (raw from environment)

    Output format matches input format.
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Handle both "image" and "images" keys
        image_key = (
            "image" if "image" in data else ("images" if "images" in data else None)
        )
        if image_key is None:
            return data

        if isinstance(data[image_key], dict):
            # Multiple images case
            for key, img in data[image_key].items():
                data[image_key][key] = self._resize_image(img)
        else:
            # Single image case
            data[image_key] = self._resize_image(data[image_key])
        return data

    def _resize_image(self, img: Union[torch.Tensor, np.ndarray, Image.Image]):
        """Resize image with padding, preserving input format (CHW tensor or HWC numpy)."""

        # Determine input format
        is_tensor = isinstance(img, torch.Tensor)
        is_chw = False

        if is_tensor:
            # Tensor: check if CHW (shape[0] == 3) or HWC (shape[-1] == 3)
            if img.dim() == 3 and img.shape[0] == 3:
                is_chw = True
                cur_height, cur_width = img.shape[1], img.shape[2]
            elif img.dim() == 3 and img.shape[-1] == 3:
                is_chw = False
                cur_height, cur_width = img.shape[0], img.shape[1]
            else:
                # Assume CHW for 3D tensors
                is_chw = True
                cur_height, cur_width = img.shape[1], img.shape[2]
            original_dtype = img.dtype
            original_device = img.device if hasattr(img, "device") else None
        elif isinstance(img, Image.Image):
            img = np.array(img)
            is_chw = False
            cur_height, cur_width = img.shape[0], img.shape[1]
            original_dtype = img.dtype
            original_device = None
        else:  # numpy array
            # Numpy: typically HWC
            if img.ndim == 3 and img.shape[0] == 3:
                is_chw = True
                cur_height, cur_width = img.shape[1], img.shape[2]
            else:
                is_chw = False
                cur_height, cur_width = img.shape[0], img.shape[1]
            original_dtype = img.dtype
            original_device = None

        # If already correct size, return as is
        if cur_height == self.height and cur_width == self.width:
            return img

        # Convert to HWC numpy for PIL resize
        if is_tensor:
            img_np = img.cpu().numpy() if hasattr(img, "cpu") else img.numpy()
        else:
            img_np = img

        if is_chw:
            img_np = np.transpose(img_np, (1, 2, 0))  # CHW -> HWC

        # Resize with padding using PIL
        resized = self._resize_with_pad_hwc(img_np, self.height, self.width)

        # Convert back to original format
        if is_chw:
            resized = np.transpose(resized, (2, 0, 1))  # HWC -> CHW

        if is_tensor:
            resized = torch.from_numpy(resized)
            if original_dtype is not None:
                resized = resized.to(dtype=original_dtype)
            if original_device is not None:
                resized = resized.to(device=original_device)
            return resized
        else:
            return resized.astype(original_dtype)

    def _resize_with_pad_hwc(
        self, img: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Resize HWC image with padding to maintain aspect ratio."""
        cur_height, cur_width = img.shape[0], img.shape[1]

        if cur_height == height and cur_width == width:
            return img

        # Calculate resize ratio maintaining aspect ratio
        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)

        # Convert to PIL for resize (PIL expects HWC uint8 or HWC float)
        if img.dtype == np.uint8:
            pil_img = Image.fromarray(img)
        else:
            # For float images, convert to uint8 for PIL, then back
            pil_img = Image.fromarray(
                (img * 255).astype(np.uint8)
                if img.max() <= 1.0
                else img.astype(np.uint8)
            )

        resized_pil = pil_img.resize(
            (resized_width, resized_height), resample=Image.BILINEAR
        )
        resized_np = np.array(resized_pil)

        # Convert back to original dtype if needed
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                resized_np = resized_np.astype(np.float32) / 255.0
            else:
                resized_np = resized_np.astype(img.dtype)

        # Create zero-padded output (HWC format)
        n_channels = img.shape[2] if img.ndim == 3 else 1
        result = np.zeros((height, width, n_channels), dtype=resized_np.dtype)

        # Center the resized image
        pad_top = (height - resized_height) // 2
        pad_left = (width - resized_width) // 2

        if resized_np.ndim == 2:
            resized_np = resized_np[..., np.newaxis]
        result[
            pad_top : pad_top + resized_height, pad_left : pad_left + resized_width
        ] = resized_np

        # Remove channel dim if original was 2D
        if img.ndim == 2:
            result = result.squeeze(-1)

        return result


class DeltaActions(DataTransformFn):
    """Converts absolute actions to delta actions relative to current state."""

    def __init__(self, mask: Optional[list[bool]]):
        self.mask = mask

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in data or self.mask is None:
            return data

        if "state" not in data:
            logger.warning(
                "Warning: DeltaActions requires 'state' but it's not present"
            )
            return data

        state = data["state"]
        actions = data["actions"]

        # Convert to tensors if needed, preserving device of actions tensor
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        device = actions.device
        dtype = actions.dtype

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=dtype, device=device)
        else:
            state = state.to(device=device, dtype=dtype)

        mask = torch.tensor(self.mask, dtype=torch.bool, device=device)
        dims = len(mask)

        # Apply delta conversion only to masked dimensions
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)  # Add time dimension

        # Subtract current state from actions for masked dimensions
        state_expanded = state[:dims].unsqueeze(0).expand(actions.shape[0], -1)
        mask_expanded = mask.unsqueeze(0).expand(actions.shape[0], -1)

        actions_copy = actions.clone()
        actions_copy[:, :dims] = torch.where(
            mask_expanded, actions[:, :dims] - state_expanded, actions[:, :dims]
        )

        data["actions"] = actions_copy
        data["state"] = state  # Preserve the converted tensor state
        return data


class AbsoluteActions(DataTransformFn):
    """Converts delta actions to absolute actions by adding current state."""

    def __init__(self, mask: Optional[list[bool]]):
        self.mask = mask

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in data or self.mask is None:
            return data

        if "state" not in data:
            logger.warning(
                "Warning: AbsoluteActions requires 'state' but it's not present"
            )
            return data

        state = data["state"]
        actions = data["actions"]

        # Convert to tensors if needed, preserving device of actions tensor
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32)
        device = actions.device
        dtype = actions.dtype

        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=dtype, device=device)
        else:
            state = state.to(device=device, dtype=dtype)

        mask = torch.tensor(self.mask, dtype=torch.bool, device=device)
        dims = len(mask)

        # Apply absolute conversion only to masked dimensions
        if len(actions.shape) == 1:
            actions = actions.unsqueeze(0)  # Add time dimension

        # Add current state to actions for masked dimensions
        state_expanded = state[:dims].unsqueeze(0).expand(actions.shape[0], -1)
        mask_expanded = mask.unsqueeze(0).expand(actions.shape[0], -1)

        actions_copy = actions.clone()
        actions_copy[:, :dims] = torch.where(
            mask_expanded, actions[:, :dims] + state_expanded, actions[:, :dims]
        )

        data["actions"] = actions_copy
        data["state"] = state  # Preserve the converted tensor state
        return data


class PromptFromLeRobotTask(DataTransformFn):
    """Extracts prompt from LeRobot dataset task following OpenPI implementation."""

    def __init__(self, tasks: dict[int, str]):
        self.tasks = tasks

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "task_index" not in data:
            raise ValueError('Cannot extract prompt without "task_index"')

        task_index = data["task_index"]
        if isinstance(task_index, torch.Tensor):
            task_index = task_index.item()
        elif isinstance(task_index, (list, tuple)):
            task_index = task_index[0]

        task_index = int(task_index)

        # Following OpenPI pattern: check if task exists, with fallback for -1
        if task_index in self.tasks:
            prompt = self.tasks[task_index]
        elif task_index == -1:
            # Handle special case of -1 (unknown task) with default prompt
            prompt = "Complete the task"
        else:
            raise ValueError(
                f"task_index={task_index} not found in task mapping: {self.tasks}"
            )

        # Return new dict with prompt added (following OpenPI pattern)
        return {**data, "prompt": prompt}


class Normalize(DataTransformFn):
    """Normalizes data using precomputed statistics.

    Supports both dict-style stats (from JSON) and NormStats objects.
    Matches OpenPI's Normalize transform behavior.
    """

    def __init__(
        self,
        norm_stats: Optional[dict[str, Any]],
        use_quantiles: bool = False,
        strict: bool = False,
        skip_dims: Optional[dict[str, list[int]]] = None,
    ):
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles
        self.strict = strict
        self.skip_dims = skip_dims or {}
        self._logged_keys: set = set()  # Track which keys we've logged

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.norm_stats is None:
            return data

        for key, stats in self.norm_stats.items():
            if key in data:
                x = data[key]
                key_skip_dims = self.skip_dims.get(key)
                if isinstance(x, np.ndarray):
                    data[key] = self._normalize(x, stats, key_skip_dims, key)
                elif isinstance(x, torch.Tensor):
                    device, dtype = x.device, x.dtype
                    x_np = x.cpu().numpy()
                    x_np = self._normalize(x_np, stats, key_skip_dims, key)
                    data[key] = torch.from_numpy(x_np).to(dtype=dtype, device=device)

        return data

    def _normalize(
        self,
        x: np.ndarray,
        stats,
        skip_dims: Optional[list[int]] = None,
        key: str = "",
    ) -> np.ndarray:
        """Normalize array using stats (supports both dict and NormStats object)."""
        if hasattr(stats, "mean"):
            mean, std = stats.mean, stats.std
            q01, q99 = getattr(stats, "q01", None), getattr(stats, "q99", None)
        else:
            mean, std = stats.get("mean"), stats.get("std")
            q01, q99 = stats.get("q01"), stats.get("q99")

        mean = np.asarray(mean, dtype=np.float32) if mean is not None else None
        std = np.asarray(std, dtype=np.float32) if std is not None else None
        q01 = np.asarray(q01, dtype=np.float32) if q01 is not None else None
        q99 = np.asarray(q99, dtype=np.float32) if q99 is not None else None

        dim = x.shape[-1]

        # Validate skip_dims don't exceed action dimension
        if skip_dims:
            invalid_dims = [d for d in skip_dims if d >= dim]
            if invalid_dims:
                raise ValueError(
                    f"skip_dims {invalid_dims} exceed data dimension {dim} for key '{key}'. "
                    f"Valid indices are 0 to {dim - 1}."
                )

        original_x = x.copy() if skip_dims else None

        if self.use_quantiles and q01 is not None and q99 is not None:
            q01 = q01[..., :dim]
            q99 = q99[..., :dim]
            result = (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        elif mean is not None and std is not None:
            mean = mean[..., :dim]
            std = std[..., :dim]
            result = (x - mean) / (std + 1e-6)
        else:
            return x

        # Restore original values for skip_dims (no normalization applied)
        if skip_dims and original_x is not None:
            for dim_idx in skip_dims:
                result[..., dim_idx] = original_x[..., dim_idx]

        # Log normalization mask (once per key)
        if key and key not in self._logged_keys:
            norm_mask = [i not in (skip_dims or []) for i in range(dim)]
            logger.info(f"Normalize '{key}' (dim={dim}): {norm_mask}")
            self._logged_keys.add(key)

        return result


class Unnormalize(DataTransformFn):
    """Unnormalizes data using precomputed statistics.

    Supports both dict-style stats (from JSON) and NormStats objects.
    Matches OpenPI's Unnormalize transform behavior.
    """

    def __init__(
        self,
        norm_stats: Optional[dict[str, Any]],
        use_quantiles: bool = False,
        skip_dims: Optional[dict[str, list[int]]] = None,
    ):
        self.norm_stats = norm_stats
        self.use_quantiles = use_quantiles
        self.skip_dims = skip_dims or {}
        self._logged_keys: set = set()  # Track which keys we've logged

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.norm_stats is None:
            return data

        for key, stats in self.norm_stats.items():
            if key in data:
                x = data[key]
                key_skip_dims = self.skip_dims.get(key)
                if isinstance(x, np.ndarray):
                    data[key] = self._unnormalize(x, stats, key_skip_dims, key)
                elif isinstance(x, torch.Tensor):
                    device, dtype = x.device, x.dtype
                    x_np = x.cpu().numpy()
                    x_np = self._unnormalize(x_np, stats, key_skip_dims, key)
                    data[key] = torch.from_numpy(x_np).to(dtype=dtype, device=device)

        return data

    def _unnormalize(
        self,
        x: np.ndarray,
        stats,
        skip_dims: Optional[list[int]] = None,
        key: str = "",
    ) -> np.ndarray:
        """Unnormalize array using stats (supports both dict and NormStats object)."""
        if hasattr(stats, "mean"):
            mean, std = stats.mean, stats.std
            q01, q99 = getattr(stats, "q01", None), getattr(stats, "q99", None)
        else:
            mean, std = stats.get("mean"), stats.get("std")
            q01, q99 = stats.get("q01"), stats.get("q99")

        mean = np.asarray(mean, dtype=np.float32) if mean is not None else None
        std = np.asarray(std, dtype=np.float32) if std is not None else None
        q01 = np.asarray(q01, dtype=np.float32) if q01 is not None else None
        q99 = np.asarray(q99, dtype=np.float32) if q99 is not None else None

        dim = x.shape[-1]

        # Validate skip_dims don't exceed action dimension
        if skip_dims:
            invalid_dims = [d for d in skip_dims if d >= dim]
            if invalid_dims:
                raise ValueError(
                    f"skip_dims {invalid_dims} exceed data dimension {dim} for key '{key}'. "
                    f"Valid indices are 0 to {dim - 1}."
                )

        original_x = x.copy() if skip_dims else None

        if self.use_quantiles and q01 is not None and q99 is not None:
            stat_dim = q01.shape[-1]
            if dim > stat_dim:
                x_norm = x[..., :stat_dim]
                x_rest = x[..., stat_dim:]
                x_unnorm = (x_norm + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
                result = np.concatenate([x_unnorm, x_rest], axis=-1)
            else:
                q01 = q01[..., :dim]
                q99 = q99[..., :dim]
                result = (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        elif mean is not None and std is not None:
            if dim > mean.shape[-1]:
                mean = np.pad(mean, (0, dim - mean.shape[-1]), constant_values=0.0)
                std = np.pad(std, (0, dim - std.shape[-1]), constant_values=1.0)
            else:
                mean = mean[..., :dim]
                std = std[..., :dim]
            result = x * (std + 1e-6) + mean
        else:
            return x

        # Restore original values for skip_dims (no unnormalization applied)
        if skip_dims and original_x is not None:
            for dim_idx in skip_dims:
                result[..., dim_idx] = original_x[..., dim_idx]

        # Log unnormalization mask (once per key)
        if key and key not in self._logged_keys:
            norm_mask = [i not in (skip_dims or []) for i in range(dim)]
            logger.info(f"Unnormalize '{key}' (dim={dim}): {norm_mask}")
            self._logged_keys.add(key)

        return result


class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    def __init__(self, model_action_dim: int):
        self.model_action_dim = model_action_dim

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        data["state"] = pad_to_dim(data["state"], self.model_action_dim, axis=-1)
        if "actions" in data:
            data["actions"] = pad_to_dim(
                data["actions"], self.model_action_dim, axis=-1
            )
        return data


def pad_to_dim(
    x: Union[torch.Tensor, np.ndarray], target_dim: int, axis: int = -1
) -> torch.Tensor:
    """Pad a tensor to the target dimension with zeros along the specified axis."""
    # Convert numpy arrays to tensors
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_size = target_dim - current_dim
        # Create padding list for torch.nn.functional.pad (works from last dim to first)
        pad_list = [0, 0] * len(x.shape)
        # Convert negative axis to positive
        if axis < 0:
            axis = len(x.shape) + axis
        # Set padding for the target axis (counting from the end)
        pad_index = (len(x.shape) - 1 - axis) * 2 + 1
        pad_list[pad_index] = pad_size

        # Create zeros with same device and dtype as input tensor
        return torch.nn.functional.pad(x, pad_list, value=0.0)
    return x


def make_bool_mask(*dims: int) -> list[bool]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == [True, True, False, False, True, True]
        make_bool_mask(2, 0, 2) == [True, True, True, True]
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * dim)
        else:
            result.extend([False] * (-dim))
    return result


def load_task_descriptions(dataset_path: Union[str, Path]) -> dict[int, str]:
    """Load task descriptions from dataset, handling multiple file formats.

    Supports:
    - tasks.jsonl: JSON Lines format with {"task_index": int, "task": str}
    - tasks.parquet: Parquet format with task descriptions as index and task_index as column
    """
    dataset_path = Path(dataset_path)
    meta_path = dataset_path / "meta"

    # Try different task file formats
    tasks_jsonl = meta_path / "tasks.jsonl"
    tasks_parquet = meta_path / "tasks.parquet"

    if tasks_jsonl.exists():
        return _load_tasks_jsonl(tasks_jsonl)
    elif tasks_parquet.exists():
        return _load_tasks_parquet(tasks_parquet)
    else:
        logger.warning(f"Warning: No task files found in {meta_path}")
        return {}


def _load_tasks_jsonl(tasks_file: Path) -> dict[int, str]:
    """Load tasks from JSON Lines format."""
    tasks = {}
    with open(tasks_file, "r") as f:
        for line in f:
            if line.strip():
                task_data = json.loads(line.strip())
                tasks[task_data["task_index"]] = task_data["task"]

    logger.info(f"Loaded {len(tasks)} task descriptions from {tasks_file}")
    return tasks


def _load_tasks_parquet(tasks_file: Path) -> dict[int, str]:
    """Load tasks from Parquet format."""
    try:
        import pyarrow.parquet as pq

        # Read parquet file
        table = pq.read_table(tasks_file)
        df = table.to_pandas()

        # In DROID format, task descriptions are the index and task_index is the column
        tasks = {}
        for task_description, row in df.iterrows():
            task_index = int(row["task_index"])  # Convert numpy int64 to Python int
            tasks[task_index] = task_description

        logger.info(f"Loaded {len(tasks)} task descriptions from {tasks_file}")
        return tasks

    except ImportError:
        logger.warning("Warning: pyarrow not available, cannot load parquet task files")
        return {}
    except Exception as e:
        logger.warning(f"Warning: Failed to load parquet task file {tasks_file}: {e}")
        return {}


def load_subtask_descriptions(dataset_path: Union[str, Path]) -> dict[str, list[str]]:
    """Load subtask descriptions from dataset.

    Expects subtasks.json in one of these locations:
    - {dataset_path}/local/{dataset_name}/meta/subtasks.json
    - {dataset_path}/meta/subtasks.json

    Format: {"task_key": ["subtask1", "subtask2", ...], ...}

    Returns:
        Dict mapping task keys to list of subtask descriptions
    """
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name

    # Try local format first (for datasets with local/ structure)
    local_subtasks = dataset_path / "local" / dataset_name / "meta" / "subtasks.json"
    meta_subtasks = dataset_path / "meta" / "subtasks.json"

    subtasks_file = None
    if local_subtasks.exists():
        subtasks_file = local_subtasks
    elif meta_subtasks.exists():
        subtasks_file = meta_subtasks

    if subtasks_file is None:
        return {}

    with open(subtasks_file, "r") as f:
        subtasks = json.load(f)

    logger.info(
        f"Loaded subtask descriptions for {len(subtasks)} tasks from {subtasks_file}"
    )
    return subtasks


def load_task_to_key_mapping(dataset_path: Union[str, Path]) -> dict[int, str]:
    """Load mapping from task index to task key.

    Expects tasks.json in one of these locations:
    - {dataset_path}/local/{dataset_name}/meta/tasks.json
    - {dataset_path}/meta/tasks.json

    Format: {"task_key": "task description", ...}
    The task index order corresponds to the order of keys in the JSON file.

    Returns:
        Dict mapping task index to task key (e.g., {0: "pick-place-0", 1: "pick-place-1"})
    """
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.name

    # Try local format first
    local_tasks = dataset_path / "local" / dataset_name / "meta" / "tasks.json"
    meta_tasks = dataset_path / "meta" / "tasks.json"

    tasks_file = None
    if local_tasks.exists():
        tasks_file = local_tasks
    elif meta_tasks.exists():
        tasks_file = meta_tasks

    if tasks_file is None:
        return {}

    with open(tasks_file, "r") as f:
        tasks = json.load(f)

    # Create mapping from index to key based on order in JSON
    return dict(enumerate(tasks.keys()))


class FASTTokenizerTransform(DataTransformFn):
    """Adds FAST action tokens to the prefix/response structure.

    This transform converts continuous actions to discrete FAST tokens and integrates
    them into the existing prefix/response pattern used for reasoning training.

    Behavior:
    1. If prefix/response already exist (e.g., from subtask reasoning):
       - prefix: unchanged
       - response: "<original_content>. FAST: <fast_tokens>"

    2. If no prefix/response exist:
       - prefix: "FAST:"
       - response: "<fast_tokens>"

    The FAST token portion has:
       - ar_mask = True (causal attention)
       - loss_mask = True (compute loss on these tokens)
       - kv_cache_mask = False (not used for continuous action generation)

    Expected input data keys:
        - actions: Normalized action tensor of shape (action_horizon, action_dim)
        - prefix (optional): Existing prefix from other transforms
        - response (optional): Existing response from other transforms

    Output data keys (added/modified):
        - prefix: Text prefix for reasoning
        - response: Text response including FAST tokens
        - fast_token_ids: Raw FAST token IDs (for decoding)
        - fast_response_start: Character position where FAST content starts in response
        - has_fast_tokens: True if FAST tokens were added
    """

    _fast_processor = None

    def __init__(
        self,
        fast_tokenizer_path: str = "physical-intelligence/fast",
    ):
        """
        Args:
            fast_tokenizer_path: HuggingFace path for FAST action tokenizer
        """
        self.fast_tokenizer_path = fast_tokenizer_path
        self._ensure_tokenizer_loaded()

    def _ensure_tokenizer_loaded(self):
        """Lazy load FAST tokenizer (shared across instances)."""
        if FASTTokenizerTransform._fast_processor is None:
            from transformers import AutoProcessor

            FASTTokenizerTransform._fast_processor = AutoProcessor.from_pretrained(
                self.fast_tokenizer_path, trust_remote_code=True
            )

    @property
    def fast_processor(self):
        return FASTTokenizerTransform._fast_processor

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = data.get("actions")

        if actions is None:
            return data

        # Convert tensors to numpy
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        # Ensure actions have batch dimension for FAST processor
        if actions.ndim == 2:
            actions_batch = actions[np.newaxis, ...]
        else:
            actions_batch = actions

        # Tokenize actions with FAST
        fast_token_ids = self.fast_processor(actions_batch)[0]

        # Convert token IDs to a string representation (space-separated integers)
        fast_token_str = " ".join(map(str, fast_token_ids))

        # Get existing prefix/response if any
        existing_prefix = data.get("prefix")
        existing_response = data.get("response")

        if existing_prefix is not None and existing_response is not None:
            # Case 1: Append FAST tokens to existing response
            # Remove trailing period/punctuation if present for cleaner formatting
            clean_response = existing_response.rstrip()
            if clean_response and clean_response[-1] in ".!?":
                fast_response_start = len(clean_response) + len(" FAST: ")
                new_response = f"{clean_response} FAST: {fast_token_str}"
            else:
                fast_response_start = len(clean_response) + len(". FAST: ")
                new_response = f"{clean_response}. FAST: {fast_token_str}"

            data["prefix"] = existing_prefix
            data["response"] = new_response
        else:
            # Case 2: Create new prefix/response with FAST tokens
            data["prefix"] = "FAST:"
            data["response"] = fast_token_str
            fast_response_start = 0

        # Store metadata for mask computation
        data["fast_token_ids"] = np.array(fast_token_ids, dtype=np.int32)
        data["fast_response_start"] = fast_response_start
        data["has_fast_tokens"] = True

        return data

    def decode_fast_tokens(
        self,
        fast_token_ids: Union[np.ndarray, list[int]],
        action_horizon: int,
        action_dim: int,
    ) -> np.ndarray:
        """Decode FAST token IDs back to continuous actions.

        Args:
            fast_token_ids: FAST token IDs (not PaliGemma vocab space)
            action_horizon: Expected number of action timesteps
            action_dim: Expected action dimensionality

        Returns:
            Decoded actions of shape (action_horizon, action_dim)
        """
        if isinstance(fast_token_ids, np.ndarray):
            fast_token_ids = fast_token_ids.tolist()

        try:
            # Suppress stderr from FAST library which prints errors before raising exceptions
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                decoded = self.fast_processor.decode(
                    [fast_token_ids],
                    time_horizon=action_horizon,
                    action_dim=action_dim,
                )
            finally:
                sys.stderr = old_stderr
            return np.array(decoded[0], dtype=np.float32)
        except Exception as e:
            logger.debug(f"FAST decode failed (expected during early training): {e}")
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

    @staticmethod
    def parse_fast_tokens_from_response(response: str) -> Optional[list[int]]:
        """Parse FAST token IDs from a response string.

        Args:
            response: Response string that may contain "FAST: <tokens>"

        Returns:
            List of FAST token IDs, or None if not found
        """
        if "FAST:" not in response:
            return None

        # Extract the FAST token portion
        fast_part = response.split("FAST:")[-1].strip()

        # Parse space-separated integers
        try:
            tokens = [int(t) for t in fast_part.split()]
            return tokens
        except ValueError:
            return None


class SubtaskReasoningTransform(DataTransformFn):
    """Injects subtask reasoning as prefix/response for CoT training.

    Uses task_idx and subtask_idx from the data to look up the current subtask
    description and adds it as prefix="Subtask:" and response=subtask_description.

    This enables training VLA models with chain-of-thought reasoning about
    the current subtask being executed.

    Expected data keys:
        - task_idx: Current task index (int or tensor)
        - subtask_idx: Current subtask index (int or tensor)

    Output:
        - prefix: "Subtask:" (or custom prefix)
        - response: Subtask description string
    """

    def __init__(
        self,
        subtasks: dict[str, list[str]],
        task_idx_to_key: dict[int, str],
        prefix: str = "Subtask:",
    ):
        """
        Args:
            subtasks: Dict mapping task keys to list of subtask descriptions
            task_idx_to_key: Dict mapping task index to task key
            prefix: Prefix string for the response (default: "Subtask:")
        """
        self.subtasks = subtasks
        self.task_idx_to_key = task_idx_to_key
        self.prefix = prefix

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        # Get task_idx and subtask_idx from data
        task_idx = data.get("task_idx")
        subtask_idx = data.get("subtask_idx")

        if task_idx is None or subtask_idx is None:
            return data

        # Convert tensors to int
        if isinstance(task_idx, torch.Tensor):
            task_idx = task_idx.item()
        elif isinstance(task_idx, (list, tuple, np.ndarray)):
            task_idx = int(task_idx[0])
        task_idx = int(task_idx)

        if isinstance(subtask_idx, torch.Tensor):
            subtask_idx = subtask_idx.item()
        elif isinstance(subtask_idx, (list, tuple, np.ndarray)):
            subtask_idx = int(subtask_idx[0])
        subtask_idx = int(subtask_idx)

        # Look up task key and subtask description
        task_key = self.task_idx_to_key.get(task_idx)
        if task_key is None:
            return data

        subtask_list = self.subtasks.get(task_key)
        if subtask_list is None or subtask_idx >= len(subtask_list):
            return data

        response = subtask_list[subtask_idx]

        return {**data, "prefix": self.prefix, "response": response}

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: Union[str, Path],
        prefix: str = "Subtask:",
    ) -> Optional["SubtaskReasoningTransform"]:
        """Create transform from dataset path.

        Args:
            dataset_path: Path to the dataset directory
            prefix: Prefix string for the response

        Returns:
            SubtaskReasoningTransform instance or None if subtasks not found
        """
        subtasks = load_subtask_descriptions(dataset_path)
        task_idx_to_key = load_task_to_key_mapping(dataset_path)

        if not subtasks or not task_idx_to_key:
            logger.warning(f"Subtask reasoning not available for {dataset_path}")
            return None

        return cls(subtasks, task_idx_to_key, prefix)


class LiberoEcotSubtaskReasoningTransform(DataTransformFn):
    """Injects subtask reasoning for Libero datasets using ECOT annotations.

    This transform uses:
      - `episode_index` and `frame_index` from each sample
      - `episodes_ecot.json` under the LeRobot dataset meta/ directory
      - `subtasks.json` files under an external ECOT data root

    It produces `prefix` and `response` fields compatible with
    `SubtaskReasoningTransform` so that the processor can apply CE loss.
    """

    def __init__(
        self,
        episodes_map: dict[int, dict[str, Any]],
        ecot_data_root: Union[str, Path],
        prefix: str = "Subtask:",
    ):
        self.episodes_map = episodes_map
        self.ecot_data_root = Path(ecot_data_root)
        self.prefix = prefix

        self._subtasks_cache: dict[tuple, Optional[list[dict[str, Any]]]] = {}
        self._missing_episodes: set = set()
        self._missing_subtask_files: set = set()
        self._no_match_frames: set = set()

    def _load_subtasks_for_episode(
        self, file_name: str, demo_id: str
    ) -> Optional[list[dict[str, Any]]]:
        key = (file_name, demo_id)
        if key in self._subtasks_cache:
            return self._subtasks_cache[key]

        subtask_path = self.ecot_data_root / file_name / demo_id / "subtasks.json"
        if not subtask_path.exists():
            if key not in self._missing_subtask_files:
                logger.warning(
                    f"[LiberoEcotSubtask] subtasks.json not found for file_name={file_name}, "
                    f"demo_id={demo_id} at {subtask_path}"
                )
                self._missing_subtask_files.add(key)
            self._subtasks_cache[key] = None
            return None

        with open(subtask_path, "r") as f:
            data = json.load(f)

        subtasks = data.get("subtasks", [])
        self._subtasks_cache[key] = subtasks
        return subtasks

    def _find_subtask_name(self, episode_index: int, frame_index: int) -> Optional[str]:
        meta = self.episodes_map.get(episode_index)
        if meta is None:
            if episode_index not in self._missing_episodes:
                self._missing_episodes.add(episode_index)
            return None

        file_name = meta.get("file_name")
        demo_id = meta.get("demo_id")
        if not file_name or not demo_id:
            return None

        subtasks = self._load_subtasks_for_episode(file_name, demo_id)
        if not subtasks:
            return None

        for sub in subtasks:
            fr = sub.get("frame_range")
            name = sub.get("subtask_name")
            if not fr or len(fr) != 2 or name is None:
                continue
            start, end = int(fr[0]), int(fr[1])
            if start <= frame_index <= end:
                return str(name)

        key = (file_name, demo_id)
        if key not in self._no_match_frames:
            logger.warning(
                f"[LiberoEcotSubtask] No subtask match for episode_index={episode_index}, "
                f"frame_index={frame_index} (file_name={file_name}, demo_id={demo_id})"
            )
            self._no_match_frames.add(key)
        return None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode_index = data.get("episode_index")
        frame_index = data.get("frame_index")
        if episode_index is None or frame_index is None:
            return data

        if isinstance(episode_index, torch.Tensor):
            episode_index = int(episode_index.item())
        elif isinstance(episode_index, (list, tuple, np.ndarray)):
            episode_index = int(episode_index[0])
        else:
            episode_index = int(episode_index)

        if isinstance(frame_index, torch.Tensor):
            frame_index = int(frame_index.item())
        elif isinstance(frame_index, (list, tuple, np.ndarray)):
            frame_index = int(frame_index[0])
        else:
            frame_index = int(frame_index)

        subtask_name = self._find_subtask_name(episode_index, frame_index)
        if subtask_name is None:
            return data

        out = dict(data)
        out["subtask_response"] = subtask_name

        existing_prefix = out.get("prefix")
        existing_response = out.get("response")

        def _append_segment(base: Optional[str], segment: str) -> str:
            base = (base or "").strip()
            segment = segment.strip()
            if not base:
                return segment
            if base[-1] not in ".!?":
                base = base + "."
            return f"{base} {segment}"

        if existing_prefix is None and existing_response is None:
            out["prefix"] = self.prefix
            out["response"] = subtask_name
        else:
            out["prefix"] = existing_prefix or self.prefix
            out["response"] = _append_segment(existing_response, subtask_name)
        return out

    @classmethod
    def from_dataset_and_ecot(
        cls,
        dataset_path: Union[str, Path],
        ecot_data_root: Union[str, Path],
        prefix: str = "Subtask:",
    ) -> Optional["LiberoEcotSubtaskReasoningTransform"]:
        """Create transform using episodes_ecot.json and ECOT data root.

        Args:
            dataset_path: Path to LeRobot dataset root (containing meta/episodes_ecot.json)
            ecot_data_root: Root directory for ECOT data (contains file_name/demo_id/subtasks.json)
            prefix: Prefix string for the response
        """
        dataset_path = Path(dataset_path)
        episodes_file = dataset_path / "meta" / "episodes_ecot.json"
        if not episodes_file.exists():
            logger.warning(
                f"[LiberoEcotSubtask] episodes_ecot.json not found at {episodes_file}; "
                "Libero ECOT subtask reasoning disabled for this dataset."
            )
            return None

        with open(episodes_file, "r") as f:
            episodes = json.load(f)

        episodes_map: dict[int, dict[str, Any]] = {}
        for rec in episodes:
            idx = rec.get("episode_index")
            if idx is None:
                continue
            episodes_map[int(idx)] = rec

        if not episodes_map:
            logger.warning(
                f"[LiberoEcotSubtask] No valid entries in episodes_ecot.json at {episodes_file}"
            )
            return None

        logger.info(
            f"Loaded ECOT episode mapping for {len(episodes_map)} episodes from {episodes_file}"
        )
        return cls(
            episodes_map=episodes_map, ecot_data_root=ecot_data_root, prefix=prefix
        )


# =============================================================================
# Helper functions for BBox/Pointing reasoning transforms
# =============================================================================


def _normalize_special_object_name(label: str) -> str:
    """Apply Libero-specific object name normalization rules."""
    if not label:
        return label

    replacements = {
        "akita black bowl": "black bowl",
        "porcelain mug": "white mug",
        "white cabinet": "cabinet",
        "wine rack": "rack",
        "wooden cabinet": "cabinet",
        "wooden two layer shelf": "shelf",
        "chefmate 8 frypan": "frying pan",
        "new salad dressing": "salad dressing",
        "red coffee mug": "red mug",
        "wooden tray": "tray",
        "black book": "book",
        "yellow book": "book",
        "desk caddy": "caddy",
        "flat stove": "stove",
    }

    out = label
    for src, dst in replacements.items():
        pattern = re.compile(re.escape(src), flags=re.IGNORECASE)
        out = pattern.sub(dst, out)
    return out


def _split_base_and_index(label: str) -> tuple[str, Optional[int]]:
    """Split an object label into base name and optional trailing integer index."""
    if not label:
        return "", None
    m = re.match(r"^(.*?)(?:\s+(\d+))?$", label.strip())
    if not m:
        return label.strip(), None
    base = m.group(1).strip()
    idx_str = m.group(2)
    idx = int(idx_str) if idx_str is not None else None
    return base, idx


def _compute_instance_positions_from_boxes(
    boxes: list[Sequence[float]],
) -> list[str]:
    """Compute relative positions (left/right/middle/front/back) for multiple instances."""
    n = len(boxes)
    if n == 0:
        return []
    if n == 1:
        return ["single"]

    centers: list[tuple[float, float]] = []
    for box in boxes:
        if len(box) >= 4:
            x1, y1, x2, y2 = box
            center_x = (float(x1) + float(x2)) / 2.0
            center_y = (float(y1) + float(y2)) / 2.0
            centers.append((center_x, center_y))
        else:
            centers.append((0.0, 0.0))

    x_coords = [c[0] for c in centers]
    y_coords = [c[1] for c in centers]
    x_range = max(x_coords) - min(x_coords) if x_coords else 0.0
    y_range = max(y_coords) - min(y_coords) if y_coords else 0.0

    is_horizontal = x_range > y_range
    positions = [""] * n

    if is_horizontal:
        sorted_indices = sorted(range(n), key=lambda i: centers[i][0])
        for rank, idx in enumerate(sorted_indices):
            if rank == 0:
                positions[idx] = "left"
            elif rank == len(sorted_indices) - 1:
                positions[idx] = "right"
            else:
                positions[idx] = "middle"
    else:
        sorted_indices = sorted(range(n), key=lambda i: centers[i][1])
        for rank, idx in enumerate(sorted_indices):
            if rank == 0:
                positions[idx] = "front"
            elif rank == len(sorted_indices) - 1:
                positions[idx] = "back"
            else:
                positions[idx] = "middle"

    return positions


def _compute_instance_positions_from_points(
    points: list[Sequence[float]],
) -> list[str]:
    """Same as `_compute_instance_positions_from_boxes`, but for 2D points."""
    n = len(points)
    if n == 0:
        return []
    if n == 1:
        return ["single"]

    coords: list[tuple[float, float]] = []
    for pt in points:
        if len(pt) >= 2:
            x, y = pt[:2]
            coords.append((float(x), float(y)))
        else:
            coords.append((0.0, 0.0))

    x_coords = [c[0] for c in coords]
    y_coords = [c[1] for c in coords]
    x_range = max(x_coords) - min(x_coords) if x_coords else 0.0
    y_range = max(y_coords) - min(y_coords) if y_coords else 0.0

    is_horizontal = x_range > y_range
    positions = [""] * n

    if is_horizontal:
        sorted_indices = sorted(range(n), key=lambda i: coords[i][0])
        for rank, idx in enumerate(sorted_indices):
            if rank == 0:
                positions[idx] = "left"
            elif rank == len(sorted_indices) - 1:
                positions[idx] = "right"
            else:
                positions[idx] = "middle"
    else:
        sorted_indices = sorted(range(n), key=lambda i: coords[i][1])
        for rank, idx in enumerate(sorted_indices):
            if rank == 0:
                positions[idx] = "front"
            elif rank == len(sorted_indices) - 1:
                positions[idx] = "back"
            else:
                positions[idx] = "middle"

    return positions


def _extract_orientation_from_subtask(subtask: str, base_name: str) -> Optional[str]:
    """Extract orientation word from subtask text for a given object base name."""
    if not subtask or not base_name:
        return None

    text = subtask.lower()
    obj = base_name.lower()

    for ori in ["left", "right", "middle", "back", "front"]:
        pattern = rf"\b{ori}\s+{re.escape(obj)}\b"
        if re.search(pattern, text):
            return ori

    phrase_to_ori = {
        "at the front": "front",
        "at the back": "back",
        "on the left": "left",
        "on the right": "right",
        "in the middle": "middle",
    }
    for phrase, ori in phrase_to_ori.items():
        pattern = rf"\b{re.escape(obj)}\s+{re.escape(phrase)}\b"
        if re.search(pattern, text):
            return ori

    return None


def _demo_id_to_episode_key(demo_id: str) -> Optional[str]:
    """Convert ECOT demo_id like 'demo_0' to episode key used in results_*.json."""
    if not demo_id:
        return None
    m = re.search(r"(\d+)$", demo_id)
    if not m:
        return None
    return m.group(1)


class LiberoEcotBBoxReasoningTransform(DataTransformFn):
    """Injects bbox reasoning strings for Libero datasets using ECOT annotations."""

    def __init__(
        self,
        episodes_map: dict[int, dict[str, Any]],
        ecot_data_root: Union[str, Path],
        bbox_data_root: Optional[Union[str, Path]],
        prefix: str = "BBox:",
    ):
        self.episodes_map = episodes_map
        self.ecot_data_root = Path(ecot_data_root)
        self.bbox_data_root = (
            Path(bbox_data_root) if bbox_data_root is not None else None
        )
        self.prefix = prefix

        self._normalize_coords: bool = False
        self._norm_width: Optional[float] = None
        self._norm_height: Optional[float] = None

        self._subtasks_cache: dict[tuple[str, str], Optional[list[dict[str, Any]]]] = {}
        self._bbox_cache: dict[tuple[str, str], Optional[dict[str, Any]]] = {}
        self._bboxes_data: Optional[dict[str, dict[str, Any]]] = None

        self._missing_episodes: set = set()
        self._missing_subtask_files: set = set()
        self._missing_bbox_episodes: set = set()
        self._no_match_frames: set = set()

    def enable_normalization(self, target_size: Optional[tuple[int, int]]) -> None:
        """Enable coordinate normalization for bbox annotations."""
        if target_size is None or len(target_size) != 2:
            return
        w, h = int(target_size[0]), int(target_size[1])
        if w <= 0 or h <= 0:
            return
        self._normalize_coords = True
        self._norm_width = float(w)
        self._norm_height = float(h)

    def _maybe_normalize_box(self, box: Sequence[float]) -> tuple[int, int, int, int]:
        """Rescale bbox from source 256x256 grid to target normalization size if enabled."""
        src_w, src_h = 256.0, 256.0
        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        if (
            not self._normalize_coords
            or self._norm_width is None
            or self._norm_height is None
        ):
            return int(x1), int(y1), int(x2), int(y2)
        sx = self._norm_width / src_w
        sy = self._norm_height / src_h
        return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)

    def _load_subtasks_for_episode(
        self, file_name: str, demo_id: str
    ) -> Optional[list[dict[str, Any]]]:
        key = (file_name, demo_id)
        if key in self._subtasks_cache:
            return self._subtasks_cache[key]

        subtask_path = self.ecot_data_root / file_name / demo_id / "subtasks.json"
        if not subtask_path.exists():
            if key not in self._missing_subtask_files:
                logger.warning(
                    f"[LiberoEcotBBox] subtasks.json not found for file_name={file_name}, "
                    f"demo_id={demo_id} at {subtask_path}"
                )
                self._missing_subtask_files.add(key)
            self._subtasks_cache[key] = None
            return None

        with open(subtask_path, "r") as f:
            data = json.load(f)

        subtasks = data.get("subtasks", [])
        self._subtasks_cache[key] = subtasks
        return subtasks

    def _find_subtask_name(self, episode_index: int, frame_index: int) -> Optional[str]:
        meta = self.episodes_map.get(episode_index)
        if meta is None:
            if episode_index not in self._missing_episodes:
                self._missing_episodes.add(episode_index)
            return None

        file_name = meta.get("file_name")
        demo_id = meta.get("demo_id")
        if not file_name or not demo_id:
            return None

        subtasks = self._load_subtasks_for_episode(file_name, demo_id)
        if not subtasks:
            return None

        for sub in subtasks:
            fr = sub.get("frame_range")
            name = sub.get("subtask_name")
            if not fr or len(fr) != 2 or name is None:
                continue
            start, end = int(fr[0]), int(fr[1])
            if start <= frame_index <= end:
                return str(name)

        if (episode_index, frame_index) not in self._no_match_frames:
            self._no_match_frames.add((episode_index, frame_index))
        return None

    def _ensure_bboxes_loaded(self):
        if self._bboxes_data is not None or self.bbox_data_root is None:
            return
        self._bboxes_data = {}
        pattern = str(self.bbox_data_root / "results_*.json")
        json_files = sorted(glob.glob(pattern))
        if not json_files:
            return

        for jf in json_files:
            with open(jf, "r") as f:
                data = json.load(f)
            for file_path, episodes in data.items():
                if file_path not in self._bboxes_data:
                    self._bboxes_data[file_path] = {}
                self._bboxes_data[file_path].update(episodes)

    def _find_bbox_episode(
        self, file_name: str, demo_id: str
    ) -> Optional[dict[str, Any]]:
        key = (file_name, demo_id)
        if key in self._bbox_cache:
            return self._bbox_cache[key]

        small_json = self.ecot_data_root / file_name / demo_id / "bounding_box.json"
        if small_json.exists():
            with open(small_json, "r") as f:
                episode_data = json.load(f)
            self._bbox_cache[key] = episode_data
            return episode_data

        self._ensure_bboxes_loaded()
        if not self._bboxes_data:
            self._bbox_cache[key] = None
            return None

        episode_key = _demo_id_to_episode_key(demo_id)
        if episode_key is None:
            self._bbox_cache[key] = None
            return None

        for data_file_path, episodes in self._bboxes_data.items():
            path_str = str(data_file_path)
            parts = Path(path_str).parts
            for i, part in enumerate(parts[:-1]):
                if part == file_name and i + 1 < len(parts) and parts[i + 1] == demo_id:
                    if episode_key in episodes:
                        episode_data = episodes[episode_key]
                        self._bbox_cache[key] = episode_data
                        return episode_data

        if key not in self._missing_bbox_episodes:
            self._missing_bbox_episodes.add(key)
        self._bbox_cache[key] = None
        return None

    def _build_information_string(
        self, episode_index: int, frame_index: int
    ) -> Optional[str]:
        meta = self.episodes_map.get(episode_index)
        if meta is None:
            if episode_index not in self._missing_episodes:
                self._missing_episodes.add(episode_index)
            return None

        file_name = meta.get("file_name")
        demo_id = meta.get("demo_id")
        if not file_name or not demo_id:
            return None

        subtask = self._find_subtask_name(episode_index, frame_index)
        if subtask is None:
            return None

        bbox_episode = self._find_bbox_episode(file_name, demo_id)
        if not bbox_episode:
            return None

        step_data_list = bbox_episode.get("bboxes_per_step", [])
        step_data = None
        for s in step_data_list:
            if int(s.get("step_idx", -1)) == frame_index:
                step_data = s
                break
        if not step_data:
            return None

        bboxes = step_data.get("bboxes", [])
        if not bboxes:
            return None

        groups: dict[str, dict[str, Any]] = {}
        for bbox in bboxes:
            label_raw = bbox.get("label", "")
            if not label_raw:
                continue
            label_norm = _normalize_special_object_name(label_raw)
            base, idx = _split_base_and_index(label_norm)
            base = base.strip()
            if not base:
                continue

            base_key = base.lower()
            if base_key not in groups:
                groups[base_key] = {"base_name": base, "boxes": [], "indices": []}
            groups[base_key]["boxes"].append(bbox.get("box", []))
            groups[base_key]["indices"].append(idx)

        if not groups:
            return None

        subtask_lower = subtask.lower()
        info_items: list[str] = []

        for base_key, data in groups.items():
            base_name = data["base_name"]
            boxes = data["boxes"]
            indices = data.get("indices", [None] * len(boxes))

            if base_key not in subtask_lower:
                if base_key.endswith(" bowl"):
                    has_generic_bowl = re.search(r"\bbowl\b", subtask_lower) is not None
                    color_words = ["black", "white"]
                    has_colored_bowl = any(
                        re.search(rf"\b{color}\s+bowl\b", subtask_lower) is not None
                        for color in color_words
                    )
                    if not (has_generic_bowl and not has_colored_bowl):
                        continue
                else:
                    continue

            if len(boxes) == 1:
                box = boxes[0]
                if len(box) >= 4:
                    x1, y1, x2, y2 = self._maybe_normalize_box(box)
                    info_items.append(f'"{base_name}": "[[{x1}, {y1}], [{x2}, {y2}]]"')
                continue

            orientation = _extract_orientation_from_subtask(subtask, base_name)
            target_indices: list[int] = []
            if orientation is None:
                idx1_candidates = [
                    i for i, numeric_idx in enumerate(indices) if numeric_idx == 1
                ]
                if idx1_candidates:
                    target_indices = idx1_candidates
                else:
                    continue
            else:
                positions = _compute_instance_positions_from_boxes(boxes)
                for idx, pos in enumerate(positions):
                    if pos == orientation:
                        target_indices.append(idx)

            for idx in target_indices:
                box = boxes[idx]
                if not box or len(box) < 4:
                    continue
                x1, y1, x2, y2 = self._maybe_normalize_box(box)
                info_items.append(f'"{base_name}": "[[{x1}, {y1}], [{x2}, {y2}]]"')

        if not info_items:
            fallback_targets: list[str] = []
            if "drawer" in subtask_lower and "cabinet" in groups:
                fallback_targets.append("cabinet")
            if "knob" in subtask_lower and "stove" in groups:
                fallback_targets.append("stove")

            for fb_key in fallback_targets:
                fb_data = groups.get(fb_key)
                if not fb_data:
                    continue
                fb_base_name = fb_data["base_name"]
                fb_boxes = fb_data["boxes"]
                for box in fb_boxes:
                    if not box or len(box) < 4:
                        continue
                    x1, y1, x2, y2 = self._maybe_normalize_box(box)
                    info_items.append(
                        f'"{fb_base_name}": "[[{x1}, {y1}], [{x2}, {y2}]]"'
                    )

        if not info_items:
            return None

        return ", ".join(info_items)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode_index = data.get("episode_index")
        frame_index = data.get("frame_index")
        if episode_index is None or frame_index is None:
            return data

        if isinstance(episode_index, torch.Tensor):
            episode_index = int(episode_index.item())
        elif isinstance(episode_index, (list, tuple, np.ndarray)):
            episode_index = int(episode_index[0])
        else:
            episode_index = int(episode_index)

        if isinstance(frame_index, torch.Tensor):
            frame_index = int(frame_index.item())
        elif isinstance(frame_index, (list, tuple, np.ndarray)):
            frame_index = int(frame_index[0])
        else:
            frame_index = int(frame_index)

        information = self._build_information_string(episode_index, frame_index)
        if information is None:
            return data

        out = dict(data)
        out["bbox_response"] = information

        existing_prefix = out.get("prefix")
        existing_response = out.get("response")
        segment = f"{self.prefix} {information}".strip()

        def _append_segment(base: Optional[str], segment_text: str) -> str:
            base = (base or "").strip()
            segment_text = segment_text.strip()
            if not base:
                return segment_text
            if base[-1] not in ".!?":
                base = base + "."
            return f"{base} {segment_text}"

        if existing_prefix is None and existing_response is None:
            out["prefix"] = self.prefix
            out["response"] = information
        else:
            out["prefix"] = existing_prefix
            out["response"] = _append_segment(existing_response, segment)
        return out

    @classmethod
    def from_dataset_and_ecot(
        cls,
        dataset_path: Union[str, Path],
        ecot_data_root: Union[str, Path],
        bbox_data_root: Optional[Union[str, Path]],
        prefix: str = "BBox:",
        normalize_coords: bool = False,
        normalization_size: Optional[tuple[int, int]] = None,
    ) -> Optional["LiberoEcotBBoxReasoningTransform"]:
        """Create transform using episodes_ecot.json, ECOT data root and bbox data root."""
        dataset_path = Path(dataset_path)
        episodes_file = dataset_path / "meta" / "episodes_ecot.json"
        if not episodes_file.exists():
            logger.warning(
                f"[LiberoEcotBBox] episodes_ecot.json not found at {episodes_file}; "
                "Libero ECOT bbox reasoning disabled for this dataset."
            )
            return None

        with open(episodes_file, "r") as f:
            episodes = json.load(f)

        episodes_map: dict[int, dict[str, Any]] = {}
        for rec in episodes:
            idx = rec.get("episode_index")
            if idx is None:
                continue
            episodes_map[int(idx)] = rec

        if not episodes_map:
            logger.warning(
                f"[LiberoEcotBBox] No valid entries in episodes_ecot.json at {episodes_file}"
            )
            return None

        logger.info(
            f"Loaded ECOT episode mapping for {len(episodes_map)} episodes from {episodes_file} (BBox)"
        )
        transform = cls(
            episodes_map=episodes_map,
            ecot_data_root=ecot_data_root,
            bbox_data_root=bbox_data_root,
            prefix=prefix,
        )
        if normalize_coords:
            transform.enable_normalization(normalization_size)
        return transform


class LiberoEcotPointingReasoningTransform(DataTransformFn):
    """Injects pointing reasoning strings for Libero datasets using ECOT annotations."""

    def __init__(
        self,
        episodes_map: dict[int, dict[str, Any]],
        ecot_data_root: Union[str, Path],
        points_data_root: Optional[Union[str, Path]],
        prefix: str = "Point:",
    ):
        self.episodes_map = episodes_map
        self.ecot_data_root = Path(ecot_data_root)
        self.points_data_root = (
            Path(points_data_root) if points_data_root is not None else None
        )
        self.prefix = prefix

        self._normalize_coords: bool = False
        self._norm_width: Optional[float] = None
        self._norm_height: Optional[float] = None

        self._subtasks_cache: dict[tuple[str, str], Optional[list[dict[str, Any]]]] = {}
        self._points_cache: dict[tuple[str, str], Optional[dict[str, Any]]] = {}
        self._points_data: Optional[dict[str, dict[str, Any]]] = None

        self._missing_episodes: set = set()
        self._missing_subtask_files: set = set()
        self._missing_points_episodes: set = set()
        self._no_match_frames: set = set()

    def enable_normalization(self, target_size: Optional[tuple[int, int]]) -> None:
        """Enable coordinate normalization for pointing annotations."""
        if target_size is None or len(target_size) != 2:
            return
        w, h = int(target_size[0]), int(target_size[1])
        if w <= 0 or h <= 0:
            return
        self._normalize_coords = True
        self._norm_width = float(w)
        self._norm_height = float(h)

    def _maybe_normalize_point(self, pt: Sequence[float]) -> tuple[int, int]:
        """Rescale point from source 256x256 grid to target normalization size if enabled."""
        src_w, src_h = 256.0, 256.0
        x, y = float(pt[0]), float(pt[1])
        if (
            not self._normalize_coords
            or self._norm_width is None
            or self._norm_height is None
        ):
            return int(x), int(y)
        sx = self._norm_width / src_w
        sy = self._norm_height / src_h
        return int(x * sx), int(y * sy)

    def _load_subtasks_for_episode(
        self, file_name: str, demo_id: str
    ) -> Optional[list[dict[str, Any]]]:
        key = (file_name, demo_id)
        if key in self._subtasks_cache:
            return self._subtasks_cache[key]

        subtask_path = self.ecot_data_root / file_name / demo_id / "subtasks.json"
        if not subtask_path.exists():
            if key not in self._missing_subtask_files:
                self._missing_subtask_files.add(key)
            self._subtasks_cache[key] = None
            return None

        with open(subtask_path, "r") as f:
            data = json.load(f)

        subtasks = data.get("subtasks", [])
        self._subtasks_cache[key] = subtasks
        return subtasks

    def _find_subtask_name(self, episode_index: int, frame_index: int) -> Optional[str]:
        meta = self.episodes_map.get(episode_index)
        if meta is None:
            if episode_index not in self._missing_episodes:
                self._missing_episodes.add(episode_index)
            return None

        file_name = meta.get("file_name")
        demo_id = meta.get("demo_id")
        if not file_name or not demo_id:
            return None

        subtasks = self._load_subtasks_for_episode(file_name, demo_id)
        if not subtasks:
            return None

        for sub in subtasks:
            fr = sub.get("frame_range")
            name = sub.get("subtask_name")
            if not fr or len(fr) != 2 or name is None:
                continue
            start, end = int(fr[0]), int(fr[1])
            if start <= frame_index <= end:
                return str(name)

        if (episode_index, frame_index) not in self._no_match_frames:
            self._no_match_frames.add((episode_index, frame_index))
        return None

    def _ensure_points_loaded(self):
        if self._points_data is not None or self.points_data_root is None:
            return
        self._points_data = {}
        pattern = str(self.points_data_root / "results_*.json")
        json_files = sorted(glob.glob(pattern))
        if not json_files:
            return

        for jf in json_files:
            with open(jf, "r") as f:
                data = json.load(f)
            for file_path, episodes in data.items():
                if file_path not in self._points_data:
                    self._points_data[file_path] = {}
                self._points_data[file_path].update(episodes)

    def _find_points_episode(
        self, file_name: str, demo_id: str
    ) -> Optional[dict[str, Any]]:
        key = (file_name, demo_id)
        if key in self._points_cache:
            return self._points_cache[key]

        small_json = self.ecot_data_root / file_name / demo_id / "pointing.json"
        if small_json.exists():
            with open(small_json, "r") as f:
                episode_data = json.load(f)
            self._points_cache[key] = episode_data
            return episode_data

        self._ensure_points_loaded()
        if not self._points_data:
            self._points_cache[key] = None
            return None

        episode_key = _demo_id_to_episode_key(demo_id)
        if episode_key is None:
            self._points_cache[key] = None
            return None

        for data_file_path, episodes in self._points_data.items():
            path_str = str(data_file_path)
            parts = Path(path_str).parts
            for i, part in enumerate(parts[:-1]):
                if part == file_name and i + 1 < len(parts) and parts[i + 1] == demo_id:
                    if episode_key in episodes:
                        episode_data = episodes[episode_key]
                        self._points_cache[key] = episode_data
                        return episode_data

        if key not in self._missing_points_episodes:
            self._missing_points_episodes.add(key)
        self._points_cache[key] = None
        return None

    def _build_information_string(
        self, episode_index: int, frame_index: int
    ) -> Optional[str]:
        meta = self.episodes_map.get(episode_index)
        if meta is None:
            if episode_index not in self._missing_episodes:
                self._missing_episodes.add(episode_index)
            return None

        file_name = meta.get("file_name")
        demo_id = meta.get("demo_id")
        if not file_name or not demo_id:
            return None

        subtask = self._find_subtask_name(episode_index, frame_index)
        if subtask is None:
            return None

        points_episode = self._find_points_episode(file_name, demo_id)
        if not points_episode:
            return None

        step_data_list = points_episode.get("points_per_step", [])
        step_data = None
        for s in step_data_list:
            if int(s.get("step_idx", -1)) == frame_index:
                step_data = s
                break
        if not step_data:
            return None

        points = step_data.get("points", [])
        if not points:
            return None

        groups: dict[str, dict[str, Any]] = {}
        for point_info in points:
            label_raw = point_info.get("label", "")
            point = point_info.get("point", [])
            if not label_raw or not point:
                continue
            label_norm = _normalize_special_object_name(label_raw)
            base, idx = _split_base_and_index(label_norm)
            base = base.strip()
            if not base:
                continue

            base_key = base.lower()
            if base_key not in groups:
                groups[base_key] = {"base_name": base, "points": [], "indices": []}
            groups[base_key]["points"].append(point)
            groups[base_key]["indices"].append(idx)

        if not groups:
            return None

        subtask_lower = subtask.lower()
        info_items: list[str] = []

        for base_key, data in groups.items():
            base_name = data["base_name"]
            pts = data["points"]
            indices = data.get("indices", [None] * len(pts))

            if base_key not in subtask_lower:
                if base_key.endswith(" bowl"):
                    has_generic_bowl = re.search(r"\bbowl\b", subtask_lower) is not None
                    color_words = ["black", "white"]
                    has_colored_bowl = any(
                        re.search(rf"\b{color}\s+bowl\b", subtask_lower) is not None
                        for color in color_words
                    )
                    if not (has_generic_bowl and not has_colored_bowl):
                        continue
                else:
                    continue

            if len(pts) == 1:
                pt = pts[0]
                if len(pt) >= 2:
                    x, y = self._maybe_normalize_point(pt)
                    info_items.append(f'"{base_name}": "[{x}, {y}]"')
                continue

            orientation = _extract_orientation_from_subtask(subtask, base_name)
            target_indices: list[int] = []
            if orientation is None:
                idx1_candidates = [
                    i for i, numeric_idx in enumerate(indices) if numeric_idx == 1
                ]
                if idx1_candidates:
                    target_indices = idx1_candidates
                else:
                    continue
            else:
                positions = _compute_instance_positions_from_points(pts)
                for idx, pos in enumerate(positions):
                    if pos == orientation:
                        target_indices.append(idx)

            for idx in target_indices:
                pt = pts[idx]
                if not pt or len(pt) < 2:
                    continue
                x, y = self._maybe_normalize_point(pt)
                info_items.append(f'"{base_name}": "[{x}, {y}]"')

        if not info_items:
            fallback_targets: list[str] = []
            if "drawer" in subtask_lower and "cabinet" in groups:
                fallback_targets.append("cabinet")
            if "knob" in subtask_lower and "stove" in groups:
                fallback_targets.append("stove")

            for fb_key in fallback_targets:
                fb_data = groups.get(fb_key)
                if not fb_data:
                    continue
                fb_base_name = fb_data["base_name"]
                fb_points = fb_data["points"]
                for pt in fb_points:
                    if not pt or len(pt) < 2:
                        continue
                    x, y = self._maybe_normalize_point(pt)
                    info_items.append(f'"{fb_base_name}": "[{x}, {y}]"')

        if not info_items:
            return None

        return ", ".join(info_items)

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        episode_index = data.get("episode_index")
        frame_index = data.get("frame_index")
        if episode_index is None or frame_index is None:
            return data

        if isinstance(episode_index, torch.Tensor):
            episode_index = int(episode_index.item())
        elif isinstance(episode_index, (list, tuple, np.ndarray)):
            episode_index = int(episode_index[0])
        else:
            episode_index = int(episode_index)

        if isinstance(frame_index, torch.Tensor):
            frame_index = int(frame_index.item())
        elif isinstance(frame_index, (list, tuple, np.ndarray)):
            frame_index = int(frame_index[0])
        else:
            frame_index = int(frame_index)

        information = self._build_information_string(episode_index, frame_index)
        if information is None:
            return data

        out = dict(data)
        out["pointing_response"] = information

        existing_prefix = out.get("prefix")
        existing_response = out.get("response")
        segment = f"{self.prefix} {information}".strip()

        def _append_segment(base: Optional[str], segment_text: str) -> str:
            base = (base or "").strip()
            segment_text = segment_text.strip()
            if not base:
                return segment_text
            if base[-1] not in ".!?":
                base = base + "."
            return f"{base} {segment_text}"

        if existing_prefix is None and existing_response is None:
            out["prefix"] = self.prefix
            out["response"] = information
        else:
            out["prefix"] = existing_prefix
            out["response"] = _append_segment(existing_response, segment)
        return out

    @classmethod
    def from_dataset_and_ecot(
        cls,
        dataset_path: Union[str, Path],
        ecot_data_root: Union[str, Path],
        points_data_root: Optional[Union[str, Path]],
        prefix: str = "Point:",
        normalize_coords: bool = False,
        normalization_size: Optional[tuple[int, int]] = None,
    ) -> Optional["LiberoEcotPointingReasoningTransform"]:
        """Create transform using episodes_ecot.json, ECOT data root and points data root."""
        dataset_path = Path(dataset_path)
        episodes_file = dataset_path / "meta" / "episodes_ecot.json"
        if not episodes_file.exists():
            logger.warning(
                f"[LiberoEcotPoint] episodes_ecot.json not found at {episodes_file}; "
                "Libero ECOT pointing reasoning disabled for this dataset."
            )
            return None

        with open(episodes_file, "r") as f:
            episodes = json.load(f)

        episodes_map: dict[int, dict[str, Any]] = {}
        for rec in episodes:
            idx = rec.get("episode_index")
            if idx is None:
                continue
            episodes_map[int(idx)] = rec

        if not episodes_map:
            logger.warning(
                f"[LiberoEcotPoint] No valid entries in episodes_ecot.json at {episodes_file}"
            )
            return None

        logger.info(
            f"Loaded ECOT episode mapping for {len(episodes_map)} episodes from {episodes_file} (Point)"
        )
        transform = cls(
            episodes_map=episodes_map,
            ecot_data_root=ecot_data_root,
            points_data_root=points_data_root,
            prefix=prefix,
        )
        if normalize_coords:
            transform.enable_normalization(normalization_size)
        return transform
