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

"""Register model-specific validation and quantization in one place."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

    from rlinf.quantization.algorithms.correction import CorrectionSpec
    from rlinf.quantization.config import TargetManifestEntry

ConfigValidator = Callable[[Mapping, "CorrectionSpec", str], Mapping[str, object]]


@dataclass(frozen=True)
class QuantizationAdapter:
    """Model operations used before FSDP wrapping and after weight synchronization.

    Install callbacks consume the resolved TargetManifestEntry sequence and require
    a backend object as a keyword argument.
    Weight loading receives the model; the runtime owns version and profile checks.
    """

    discover: Callable[[nn.Module], Sequence[TargetManifestEntry]]
    apply_actor: Callable[..., Sequence[TargetManifestEntry]]
    apply_rollout: Callable[..., Sequence[TargetManifestEntry]]
    load_rollout_weights: Callable[..., None]


@dataclass(frozen=True)
class QuantizationModel:
    """CPU config validation paired with a lazy execution adapter factory."""

    validate_config: ConfigValidator
    build_adapter: Callable[[], QuantizationAdapter]


_MODEL_REGISTRY: dict[str, QuantizationModel] = {}


def register_quantization_model(
    model_type: str,
    validate_config: ConfigValidator,
    build_adapter: Callable[[], QuantizationAdapter],
) -> None:
    """Register both hooks; factories must defer model and backend imports."""
    if (
        not isinstance(model_type, str)
        or not model_type
        or model_type in _MODEL_REGISTRY
    ):
        raise ValueError(
            f"quantization model already registered or invalid: {model_type!r}"
        )
    if not callable(validate_config) or not callable(build_adapter):
        raise TypeError(
            "quantization model validation and adapter factory must be callable"
        )
    _MODEL_REGISTRY[model_type] = QuantizationModel(validate_config, build_adapter)


def get_quantization_model(model_type: str) -> QuantizationModel:
    """Return one complete model entry, without constructing its execution adapter."""
    if not isinstance(model_type, str) or model_type not in _MODEL_REGISTRY:
        raise ValueError(
            f"no quantization model registered for model_type={model_type!r}"
        )
    return _MODEL_REGISTRY[model_type]


def _validate_openpi(
    model: Mapping, correction: CorrectionSpec, scope: str
) -> Mapping[str, object]:
    from rlinf.quantization.models.openpi.config import validate_pi05_model_config

    return validate_pi05_model_config(model, correction, scope)


def _build_openpi_adapter() -> QuantizationAdapter:
    from rlinf.quantization.models.openpi.adapter import build_adapter

    return build_adapter()


register_quantization_model("openpi", _validate_openpi, _build_openpi_adapter)
