# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from typing import Any


class _IndexRecordingDataset:
    """Transparent proxy that records indices sent to a map-style dataset."""

    def __init__(self, dataset: Any):
        self._dataset = dataset
        self.requested_indices: list[list[int]] = []

    def __getitem__(self, index: Any) -> Any:
        self.requested_indices.append(self._normalize_indices(index))
        return self._dataset[index]

    def __len__(self) -> int:
        return len(self._dataset)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    @staticmethod
    def _normalize_indices(index: Any) -> list[int]:
        """Represent scalar or vector dataset indexing as a list of Python ints."""
        value = index.tolist() if hasattr(index, "tolist") else index
        if isinstance(value, (tuple, list)):
            return [int(item) for item in value]
        return [int(value)]
