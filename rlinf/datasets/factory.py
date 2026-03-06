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

"""Dataset factory with registry for unified dataset creation."""

from typing import Callable, Optional


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
