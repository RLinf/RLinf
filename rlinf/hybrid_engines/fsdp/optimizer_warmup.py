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

from torch.nn import Module, Parameter


class OptimizerWarmup:
    """Track critic warmup and own the parameters frozen for it."""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.optimizer_steps = 0
        self._frozen_parameters: list[Parameter] = []

    @property
    def active(self) -> bool:
        """Return whether the current update trains only the critic."""
        return self.optimizer_steps < self.steps

    def apply(self, model: Module) -> None:
        """Apply the current warmup phase to the model's actor parameters."""
        if self.active:
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and "value_head" not in name:
                    parameter.requires_grad_(False)
                    self._frozen_parameters.append(parameter)
        else:
            for parameter in self._frozen_parameters:
                parameter.requires_grad_(True)
            self._frozen_parameters.clear()

    def restore(self, model: Module, *, optimizer_steps: int, steps: int) -> bool:
        """Restore progress and return whether the optimizer groups must change."""
        was_active = self.active
        self.optimizer_steps = optimizer_steps
        self.steps = steps
        changed = self.active != was_active
        if changed:
            self.apply(model)
        return changed
