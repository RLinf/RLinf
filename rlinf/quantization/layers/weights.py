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

"""Nonpersistent quantized weights derived from externally owned master parameters."""

from __future__ import annotations

import torch
from torch import nn

from rlinf.quantization.backends.base import QuantizationBackend, QuantizedTensor


class QuantizedWeight(nn.Module):
    """Own derived buffers; the policy runtime owns versions and model readiness."""

    def __init__(self, backend: QuantizationBackend) -> None:
        super().__init__()
        self.backend = backend
        self.logical_shape: tuple[int, ...] | None = None
        self.padded_shape: tuple[int, int] | None = None
        for name in ("data", "scales", "global_scale"):
            self.register_buffer(name, None, persistent=False)

    @property
    def is_loaded(self) -> bool:
        return self.logical_shape is not None

    @property
    def value(self) -> QuantizedTensor:
        """Build a view from current buffers, including after buffer replacement."""
        if not self.is_loaded:
            raise RuntimeError("quantized weights are not loaded")
        return QuantizedTensor(
            self.data,
            self.scales,
            self.logical_shape,
            self.backend.quantization_format,
            storage="native",
            global_scale=self.global_scale,
            padded_shape=self.padded_shape,
        )

    def clear(self) -> None:
        """Make the value unavailable while retaining allocations for the next load."""
        self.logical_shape = self.padded_shape = None

    def _apply(self, fn, recurse=True):
        self.clear()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear()
        return super()._load_from_state_dict(*args, **kwargs)

    @torch.no_grad()
    def load_weights(self, master_weight: torch.Tensor) -> None:
        """Quantize master weights and expose the value only after all buffers update."""
        self.clear()
        value = self.backend.quantize(master_weight, role="weight", native=True)
        for name in ("data", "scales", "global_scale"):
            source, current = getattr(value, name), getattr(self, name)
            if (
                source is not None
                and current is not None
                and current.shape == source.shape
                and current.device == source.device
                and current.dtype == source.dtype
            ):
                current.copy_(source)
            else:
                setattr(self, name, source)
        self.logical_shape, self.padded_shape = value.logical_shape, value.padded_shape
