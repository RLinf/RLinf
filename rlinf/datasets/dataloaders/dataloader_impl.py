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
DataLoader implementations for various dataset types.

This module provides lightweight DataLoader wrappers that expose data_config()
and yield batches in the expected format for different model types.
"""

from typing import Any, Iterator

from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    Observation as CFGObservation,
)


class CFGDataLoaderImpl:
    """DataLoader wrapper for CFG_MODEL with guidance prompts.

    Yields (CFGObservation, actions) tuples where CFGObservation contains
    positive and negative guidance prompt tokens.
    """

    def __init__(self, data_config, data_loader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self):
        return self._data_config

    def __len__(self):
        return len(self._data_loader)

    def set_epoch(self, epoch: int):
        """Forward set_epoch to sampler and dataset for proper shuffling each epoch."""
        if hasattr(self._data_loader.sampler, "set_epoch"):
            self._data_loader.sampler.set_epoch(epoch)
        if hasattr(self._data_loader.dataset, "set_epoch"):
            self._data_loader.dataset.set_epoch(epoch)

    def __iter__(self):
        for batch in self._data_loader:
            # Use rlinf's CFGObservation.from_dict, which includes positive/negative guidance prompts
            yield CFGObservation.from_dict(batch), batch["actions"]


class VlaLibValueModelDataLoaderImpl:
    """DataLoader wrapper for vla_lib ValueDataset / ValueMixtureDataset.

    Yields batch dicts directly as returned by the underlying dataset.
    """

    def __init__(self, data_config, data_loader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self):
        return self._data_config

    def __len__(self):
        return len(self._data_loader)

    def set_epoch(self, epoch: int):
        """Forward set_epoch to sampler and dataset for proper shuffling each epoch."""
        if hasattr(self._data_loader.sampler, "set_epoch"):
            self._data_loader.sampler.set_epoch(epoch)
        if hasattr(self._data_loader.dataset, "set_epoch"):
            self._data_loader.dataset.set_epoch(epoch)

    def __iter__(self):
        for batch in self._data_loader:
            yield batch


class ValueDataLoaderImpl:
    """Lightweight wrapper that yields batches and exposes data_config().

    This is the generic value model dataloader used by FSDPValueSftWorker.
    """

    def __init__(self, data_config: dict, data_loader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> dict:
        return self._data_config

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        """Forward set_epoch to sampler and dataset for proper shuffling each epoch."""
        if hasattr(self._data_loader.sampler, "set_epoch"):
            self._data_loader.sampler.set_epoch(epoch)
        if hasattr(self._data_loader.dataset, "set_epoch"):
            self._data_loader.dataset.set_epoch(epoch)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self._data_loader
