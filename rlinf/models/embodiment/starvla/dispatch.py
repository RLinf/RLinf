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

"""Dispatch starVLA forward/rollout handlers based on action-head type."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from .default_forward.adapter import run_default_forward_adapter
from .default_forward.fast import run_default_forward_fast
from .default_forward.flowmatching import run_default_forward_flowmatching
from .default_forward.oft import run_default_forward_oft
from .rollout.adapter import run_rollout_adapter
from .rollout.fast import run_rollout_fast
from .rollout.flowmatching import run_rollout_flowmatching
from .rollout.oft import run_rollout_oft

DefaultForwardHandler = Callable[..., dict[str, torch.Tensor | None]]
RolloutHandler = Callable[..., dict[str, Any]]

DEFAULT_FORWARD_HANDLERS = {
    "fast": run_default_forward_fast,
    "oft": run_default_forward_oft,
    "adapter": run_default_forward_adapter,
    "pi": run_default_forward_flowmatching,
    "gr00t": run_default_forward_flowmatching,
    "dual": run_default_forward_flowmatching,
}

ROLLOUT_HANDLERS = {
    "fast": run_rollout_fast,
    "oft": run_rollout_oft,
    "adapter": run_rollout_adapter,
    "pi": run_rollout_flowmatching,
    "gr00t": run_rollout_flowmatching,
    "dual": run_rollout_flowmatching,
}


def get_default_forward_handler(action_head_name: str) -> DefaultForwardHandler | None:
    """Return the training-time default_forward handler for a given action head."""
    return DEFAULT_FORWARD_HANDLERS.get(action_head_name)


def get_rollout_handler(action_head_name: str) -> RolloutHandler | None:
    """Return the rollout-time handler for a given action head."""
    return ROLLOUT_HANDLERS.get(action_head_name)
