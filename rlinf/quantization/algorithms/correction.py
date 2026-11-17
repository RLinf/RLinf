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

"""Configuration and argument routing for PPO behavior correction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class CorrectionSpec:
    mode: str = "off"
    max_weight: float = 2.0
    reject_weight_above: float | None = None


def parse_correction_spec(value: Mapping | None) -> CorrectionSpec:
    """Parse loss-correction settings independently of the numerical quantization recipe."""
    value = {} if value is None else value
    if not isinstance(value, Mapping):
        raise ValueError("qat.correction must be a mapping")
    unknown = set(value) - {"mode", "max_weight", "reject_weight_above"}
    if unknown:
        raise ValueError(f"unknown correction fields: {sorted(unknown)}")
    mode = value.get("mode", "off")
    if mode not in {"off", "importance_sampling"}:
        raise ValueError(f"unsupported correction mode: {mode!r}")
    cap = value.get("max_weight", 2.0)
    threshold = value.get("reject_weight_above")
    for name, number in (("max_weight", cap), ("reject_weight_above", threshold)):
        if number is None and name == "reject_weight_above":
            continue
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or number <= 0
        ):
            raise ValueError(f"correction.{name} must be finite and positive")
    return CorrectionSpec(
        mode, float(cap), None if threshold is None else float(threshold)
    )


def correction_loss_kwargs(batch: Mapping, spec: CorrectionSpec) -> dict:
    """Keep rollout scores and pass the separate frozen anchor to decoupled PPO."""
    frozen = batch["proximal_logprobs"].detach().float()
    behavior = batch["prev_logprobs"].detach().float()
    return {
        "old_logprobs": behavior if spec.mode == "importance_sampling" else frozen,
        "proximal_logprobs": frozen,
        "behave_weight_clip_max": spec.max_weight
        if spec.mode == "importance_sampling"
        else None,
        "behave_weight_threshold": spec.reject_weight_above
        if spec.mode == "importance_sampling"
        else None,
    }
