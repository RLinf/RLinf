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

"""Config helpers for the prefix off-policy runtime."""

from typing import Any

from omegaconf import OmegaConf

PREFIX_OFF_POLICY_LOSS_TYPE = "prefix_off_policy"


def is_prefix_off_policy_loss(loss_type: Any) -> bool:
    """Return whether a config selects the prefix off-policy runtime."""
    return str(loss_type) == PREFIX_OFF_POLICY_LOSS_TYPE


def is_gate_trace_enabled(cfg: Any) -> bool:
    """Return whether rollout diagnostics are collected for gate calibration."""
    return bool(
        OmegaConf.select(cfg, "algorithm.rlt_gate_calibration.enable", default=False)
    )
