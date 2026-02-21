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

"""Unified base interface for all dataset types."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from torch.utils.data import Dataset


class UnifiedDatasetInterface(Dataset, ABC):
    """Base interface for VLM, VLA, and RLDS datasets."""

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        pass

    def get_train_val_split(
        self, validation_split: Optional[float] = None
    ) -> tuple["UnifiedDatasetInterface", "UnifiedDatasetInterface"]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support train/val splitting"
        )

    def get_custom_tokens(self) -> Optional[list[str]]:
        return None

    def get_source_name(self) -> str:
        return self.__class__.__name__.lower().replace("dataset", "")
