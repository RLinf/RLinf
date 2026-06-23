# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-environment eval pre/post-processing interface for ``openpi_pytorch``.

An :class:`EvalProcessor` turns raw env observations into a model
``Observation`` and maps sampled model actions back into env actions. Each
environment provides a concrete subclass (e.g. ``BehaviorEvalProcessor``); the
model factory selects one by env name via
:func:`rlinf.data.datasets.openpi_pytorch.get_eval_processor`, so the factory is
not coupled to any single environment.

This module is intentionally dependency-free (no torch / model imports) so it can
be imported cheaply when resolving the registry. It is a thin interface rather
than an ``abc.ABC`` so subclasses introduce no metaclass change.
"""

from __future__ import annotations


class EvalProcessor:
    """Interface for env-specific openpi_pytorch eval pre/post-processing."""

    def build_observation(self, env_obs: dict, device):
        """Turn a raw env observation dict into a model ``Observation``."""
        raise NotImplementedError

    def postprocess_actions(self, model_actions):
        """Map sampled model actions back into env actions."""
        raise NotImplementedError
