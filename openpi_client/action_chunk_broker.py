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
"""Compatibility action chunk broker for OpenPI policies."""

from __future__ import annotations

from typing import Any


class ActionChunkBroker:
    """Executes a fixed number of actions from each policy action chunk."""

    def __init__(self, policy: Any, action_horizon: int) -> None:
        """Initialize the broker.

        Args:
            policy: Policy object with an ``infer`` method returning ``actions``.
            action_horizon: Number of actions to consume before refreshing a chunk.
        """
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._policy = policy
        self._action_horizon = action_horizon
        self._chunk_outputs: dict[str, Any] | None = None
        self._chunk_index = 0

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Return the next single action from the current policy chunk."""
        if self._chunk_outputs is None or self._chunk_index >= self._action_horizon:
            self._chunk_outputs = dict(self._policy.infer(obs))
            self._chunk_index = 0

        if "actions" not in self._chunk_outputs:
            raise KeyError("policy output must contain an 'actions' entry")

        actions = self._chunk_outputs["actions"]
        if self._chunk_index >= len(actions):
            raise IndexError(
                "policy returned fewer actions than the configured action_horizon"
            )

        output = dict(self._chunk_outputs)
        output["actions"] = actions[self._chunk_index]
        self._chunk_index += 1
        return output
