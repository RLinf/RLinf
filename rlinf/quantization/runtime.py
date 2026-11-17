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

"""Shared numerical recipes and transactional publication of quantized weights."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import torch
from torch import nn

from rlinf.quantization.backends import create_quantization_backend
from rlinf.quantization.config import (
    ResolvedProfile,
    resolve_quantization_profile,
    validate_qat_config,
)
from rlinf.quantization.layers.weights import QuantizedWeight
from rlinf.quantization.registry import get_quantization_model


class QuantizedPolicyRuntime:
    """Own model readiness and publish one version after every weight loads."""

    def __init__(
        self,
        model: nn.Module,
        profile: ResolvedProfile,
        stage: Literal["actor", "rollout"],
        load_rollout_weights: Callable[..., None],
    ) -> None:
        self.model = model
        self.profile = profile
        self.profile_hash = profile.config_hash
        self.stage = stage
        self._load_rollout_weights = load_rollout_weights
        self._ready = False
        self.version: int | None = None
        self._quantized_weights: tuple[QuantizedWeight, ...] = ()

    def _validate_weight_recipe(self, weight: QuantizedWeight) -> None:
        if weight.backend.quantization_format != self.profile.spec.weight.format:
            raise RuntimeError("quantized weight format does not match profile")
        if weight.backend.block_size != self.profile.spec.activation.block_size:
            raise RuntimeError("quantized weight block size does not match profile")

    def invalidate(self) -> None:
        """Withdraw the model version and every exposed quantized weight value."""
        self._ready = False
        self.version = None
        self._quantized_weights = tuple(
            module
            for module in self.model.modules()
            if isinstance(module, QuantizedWeight)
        )
        for weight in self._quantized_weights:
            weight.clear()
        self.model._quantization_weight_version = None
        self.model._quantization_profile_hash = None

    def load_weights(self, version: int) -> None:
        """Rebuild all derived weights and publish only after complete success."""
        self.invalidate()
        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError(
                "quantization weight version must be a nonnegative integer"
            )
        try:
            if self.stage == "rollout":
                weights = self._quantized_weights
                if not weights:
                    raise RuntimeError(
                        "native quantization runtime has no quantized weights"
                    )
                self._load_rollout_weights(self.model)
                for weight in weights:
                    self._validate_weight_recipe(weight)
                    if not weight.is_loaded:
                        raise RuntimeError("quantized weights are not loaded")
            self.version = version
            self.model._quantization_profile_hash = self.profile_hash
            self.model._quantization_weight_version = version
            self._ready = True
        except BaseException:
            self.invalidate()
            raise

    def assert_ready(self) -> None:
        """Check whether the complete model version is ready for sampling."""
        if not self._ready:
            raise RuntimeError("quantization runtime is not ready for sampling")


def configure_quantized_policy(
    model: nn.Module,
    cfg: Mapping[str, Any],
    stage: Literal["actor", "rollout"],
    device: torch.device | str | int,
) -> QuantizedPolicyRuntime | None:
    spec = validate_qat_config(cfg)
    if not spec.enabled:
        return None
    if stage not in ("actor", "rollout"):
        raise ValueError(
            f"quantization stage must be actor or rollout, found {stage!r}"
        )
    if getattr(model, "_quantization_runtime", None) is not None:
        raise ValueError("quantization policy is already configured")
    model_type = cfg["actor"]["model"]["model_type"]
    backend = create_quantization_backend(
        spec.weight.format,
        block_size=spec.activation.block_size,
    )
    capability = backend.probe(device)
    adapter = get_quantization_model(model_type).build_adapter()
    targets = adapter.discover(model)
    profile = resolve_quantization_profile(
        spec, [target.path for target in targets], capability, target_manifest=targets
    )
    keys = tuple(model.state_dict())
    parameters = dict(model.named_parameters())
    if stage == "actor":
        selected = adapter.apply_actor(model, profile.target_manifest, backend=backend)
    else:
        selected = adapter.apply_rollout(
            model,
            profile.target_manifest,
            backend=backend,
        )
    if tuple(target.path for target in selected) != profile.targets:
        raise RuntimeError(
            "quantization adapter selected layers differ from resolved manifest"
        )
    if tuple(model.state_dict()) != keys or any(
        dict(model.named_parameters()).get(key) is not parameter
        for key, parameter in parameters.items()
    ):
        raise RuntimeError(
            "quantization adapter changed master parameter identity or checkpoint keys"
        )
    runtime = QuantizedPolicyRuntime(
        model, profile, stage, adapter.load_rollout_weights
    )
    model._quantization_runtime = runtime
    runtime.load_weights(0)
    return runtime
