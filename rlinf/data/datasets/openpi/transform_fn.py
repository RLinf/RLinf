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

"""JAX-free ``DataTransformFn`` and ``compose``.

Copied from ``openpi.transforms``: that module imports jax and flax at the
top even for the Protocol and the one-line compose helper. Dedicated SFT
loaders only need these two symbols, so they live here instead of pulling
the openpi package at import time.

The composed callable is duck-typed. Runtime pipelines can still mix in
openpi transform objects built later by ``build_openpi_transforms``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, TypeAlias, runtime_checkable

DataDict: TypeAlias = Mapping[str, Any]


@runtime_checkable
class DataTransformFn(Protocol):
    """A per-sample transform over an unbatched nested dict."""

    def __call__(self, data: DataDict) -> DataDict:
        """Apply the transform.

        Args:
            data: Possibly nested dict of unbatched numpy leaves.

        Returns:
            The transformed data, either ``data`` mutated in place or a new
            mapping.
        """


@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """Apply a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)
