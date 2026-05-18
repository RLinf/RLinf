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

import copy
from typing import Any

import numpy as np
import torch


class BootstrapResetPlannerMixin:
    """Mixin for environments that participate in env double-buffer bootstrap planning.

    Provides the plan/apply protocol so that the active pool can snapshot its
    planner state (reseeded reset IDs, start offset, generator state) and the
    standby pool can restore it for a deterministic hand-off at epoch boundaries.

    Set ``_manages_bootstrap_state = False`` on subclasses that do not own
    bootstrap state (e.g. stateless procedural envs). In that case
    ``plan_next_bootstrap_reset`` returns ``None`` and the standby pool resets
    independently, providing latency hiding without state continuity.
    """

    bootstrap_planner_array_fields = ("reset_state_ids",)
    bootstrap_planner_scalar_fields = ("start_idx",)
    bootstrap_planner_generator_fields = (("generator_state", "_generator"),)
    supports_env_double_buffer: bool = True
    _manages_bootstrap_state: bool = True

    def _clone_bootstrap_array_field(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone().cpu()
        return np.array(value, copy=True)

    def _restore_bootstrap_array_field(self, value):
        if isinstance(value, torch.Tensor):
            device = getattr(self, "device", None)
            return value.clone().to(device) if device is not None else value.clone()
        return np.array(value, copy=True)

    def _snapshot_bootstrap_generator_state(self, generator):
        if isinstance(generator, torch.Generator):
            return generator.get_state()
        return copy.deepcopy(generator.bit_generator.state)

    def _restore_bootstrap_generator_state(self, generator, state) -> None:
        if isinstance(generator, torch.Generator):
            generator.set_state(state)
            return
        generator.bit_generator.state = copy.deepcopy(state)

    def _snapshot_bootstrap_planner_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        for field_name in self.bootstrap_planner_array_fields:
            state[field_name] = self._clone_bootstrap_array_field(
                getattr(self, field_name)
            )
        for field_name in self.bootstrap_planner_scalar_fields:
            if hasattr(self, field_name):
                state[field_name] = int(getattr(self, field_name))
        for state_key, attr_name in self.bootstrap_planner_generator_fields:
            generator = getattr(self, attr_name)
            state[state_key] = self._snapshot_bootstrap_generator_state(generator)
        return state

    def _restore_bootstrap_planner_state(self, state: dict[str, Any]) -> None:
        for field_name in self.bootstrap_planner_array_fields:
            setattr(
                self,
                field_name,
                self._restore_bootstrap_array_field(state[field_name]),
            )
        for field_name in self.bootstrap_planner_scalar_fields:
            if field_name in state:
                setattr(self, field_name, int(state[field_name]))
        for state_key, attr_name in self.bootstrap_planner_generator_fields:
            generator = getattr(self, attr_name)
            self._restore_bootstrap_generator_state(generator, state[state_key])

    def plan_next_bootstrap_reset(self) -> dict[str, Any] | None:
        """Produce a plan dict with the next reset IDs and a snapshot of planner state."""
        if not self._manages_bootstrap_state:
            return None
        if not (self.use_fixed_reset_state_ids and self.is_start):
            self.update_reset_state_ids()
        return {"planner_state": self._snapshot_bootstrap_planner_state()}

    def apply_bootstrap_plan(self, plan):
        """Restore planner state and return reset_state_ids for subsequent reset."""
        planner_state = None if plan is None else plan.get("planner_state", None)
        if planner_state is not None:
            self._restore_bootstrap_planner_state(planner_state)
            return self.reset_state_ids
        return None
