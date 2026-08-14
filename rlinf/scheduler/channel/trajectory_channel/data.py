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

from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from rlinf.data.schema.embodied_types import (
    EmbodiedRolloutResult,
    EnvResult,
    TrajectorySource,
    put_tensor_device,
)


@dataclass(kw_only=True)
class PolicyStep:
    """Policy inference associated with one or more environment sources."""

    sources: list[TrajectorySource]
    obs: dict[str, Any]
    rollout_result: EmbodiedRolloutResult

    def __post_init__(self) -> None:
        """Move inference payloads to CPU for transport."""
        self.obs = put_tensor_device(self.obs, "cpu")


@dataclass(kw_only=True)
class DummyPolicyStep:
    """Action chunk supplied without policy inference."""

    sources: list[TrajectorySource]
    obs: dict[str, Any]
    actions: torch.Tensor

    def __post_init__(self) -> None:
        """Move the externally supplied action payload to CPU."""
        self.obs = put_tensor_device(self.obs, "cpu")
        self.actions = self.actions.cpu().contiguous()


@dataclass(kw_only=True)
class EnvStepResult:
    """Completed environment outcome for policy actions."""

    sources: list[TrajectorySource]
    result: EnvResult
    next_obs: dict[str, Any]
    forward_inputs: dict[str, Any] | None
    bootstrap_values: torch.Tensor | None
    final_prev_values: torch.Tensor | None

    def __post_init__(self) -> None:
        """Move model-derived completion data to CPU for transport."""
        self.next_obs = put_tensor_device(self.next_obs, "cpu")
        if self.forward_inputs is not None:
            self.forward_inputs = put_tensor_device(self.forward_inputs, "cpu")
        if self.bootstrap_values is not None:
            self.bootstrap_values = self.bootstrap_values.cpu().contiguous()
        if self.final_prev_values is not None:
            self.final_prev_values = self.final_prev_values.cpu().contiguous()


@dataclass(kw_only=True)
class TrajectoryStart:
    """Initial environment state for one trajectory source and epoch."""

    source: TrajectorySource
    result: EnvResult


TrajectoryData: TypeAlias = (
    PolicyStep | DummyPolicyStep | TrajectoryStart | EnvStepResult
)
