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
PyTorch implementation of OpenPI transforms for dataset preprocessing.

This module provides PyTorch-based transforms that match the functionality
of the original JAX-based OpenPI transforms, designed to work with
HuggingFace datasets and the broader PyTorch ecosystem.
"""

import json
import logging
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
