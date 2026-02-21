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

"""Dataset factory with registry for unified dataset creation."""

import re
from pathlib import Path
from typing import Callable, Optional

from .base_interface import UnifiedDatasetInterface


class DatasetRegistry:
    """Registry for all dataset types."""

    _registry: dict[str, Callable] = {}
    _type_patterns: dict[str, list] = {}

    @classmethod
    def register(
        cls,
        dataset_type: str,
        creator_fn: Callable,
        path_patterns: Optional[list] = None,
    ):
        cls._registry[dataset_type] = creator_fn
        if path_patterns:
            cls._type_patterns[dataset_type] = path_patterns

    @classmethod
    def create(cls, dataset_type: str, **kwargs) -> UnifiedDatasetInterface:
        if dataset_type not in cls._registry:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. Available: {list(cls._registry.keys())}"
            )
        return cls._registry[dataset_type](**kwargs)

    @classmethod
    def detect_type(cls, dataset_path: str) -> Optional[str]:
        path_lower = dataset_path.lower()
        for dataset_type, patterns in cls._type_patterns.items():
            for pattern in patterns:
                if isinstance(pattern, str) and pattern in path_lower:
                    return dataset_type
                elif hasattr(pattern, "search") and pattern.search(dataset_path):
                    return dataset_type
        return None

    @classmethod
    def list_types(cls) -> list:
        return list(cls._registry.keys())


def register_dataset(dataset_type: str, path_patterns: Optional[list] = None):
    """Decorator to register a dataset class or creator function."""

    def decorator(target):
        if isinstance(target, type):

            def creator_fn(**kwargs):
                return target(**kwargs)

            DatasetRegistry.register(dataset_type, creator_fn, path_patterns)
            return target
        else:
            DatasetRegistry.register(dataset_type, target, path_patterns)
            return target

    return decorator


def create_dataset(
    dataset_path: str, dataset_type: Optional[str] = None, **kwargs
) -> UnifiedDatasetInterface:
    """Create a dataset with automatic type detection."""
    if dataset_type is None:
        dataset_type = DatasetRegistry.detect_type(dataset_path)
        if dataset_type is None:
            raise ValueError(
                f"Could not detect dataset type for: {dataset_path}. "
                f"Available types: {DatasetRegistry.list_types()}"
            )
    return DatasetRegistry.create(dataset_type, dataset_path=dataset_path, **kwargs)


def detect_dataset_source_name(
    dataset_path: str, dataset_type: Optional[str] = None
) -> str:
    """Detect a readable dataset source name."""
    dataset_name_lower = dataset_path.lower()

    if "robo2vlm" in dataset_name_lower:
        return "robo2vlm"
    if any(
        kw in dataset_name_lower
        for kw in ["embodied_vqa", "embodied-vqa", "embodiedvqa"]
    ):
        return "embodied_vqa"
    if dataset_type and "/" in dataset_type:
        return dataset_type.split("/")[1]

    basename = Path(dataset_path).name
    return re.sub(r"[^a-zA-Z0-9]+", "_", basename).strip("_").lower() or "dataset"
